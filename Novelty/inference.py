import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

# Constants
ALL_PLAYERS = ["england", "france", "germany", "italy", "austria", "russia", "turkey"]
PLAYER_TO_IDX = {name: i for i, name in enumerate(ALL_PLAYERS)}
NUM_PLAYERS = len(ALL_PLAYERS)
PLAYER_PAIRS = sorted([tuple(sorted(pair)) for pair in __import__('itertools').combinations(ALL_PLAYERS, 2)])
PAIR_TO_EDGE_IDX = {pair: i for i, pair in enumerate(PLAYER_PAIRS)}
NUM_EDGES = len(PLAYER_PAIRS)
SEASONS = ["fall", "winter", "spring"]
SEASON_TO_IDX = {season: i for i, season in enumerate(SEASONS)}
MIN_YEAR = 1901
MAX_YEAR = 1909
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
SCORE_IMPUTATION_VALUE = 3.0


# Model Classes
class RelationshipLSTM(torch.nn.Module):
    def __init__(self, input_dim=NUM_PLAYERS + 3, hidden_dim=64, output_dim=NUM_EDGES, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        lstm_out, (hn, cn) = self.lstm(sequence)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out


class DiplomacyGraphModel(torch.nn.Module):
    def __init__(self, bert_model_name=BERT_MODEL_NAME, lstm_hidden_dim=64, dropout_rate=0.1, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.relationship_lstm = RelationshipLSTM(input_dim=NUM_PLAYERS + 3, hidden_dim=lstm_hidden_dim)
        bert_output_dim = self.bert.config.hidden_size
        scores_dim = NUM_PLAYERS
        relationship_feature_dim = 1
        timestamp_feature_dim = 1
        combined_feature_dim = bert_output_dim + scores_dim + relationship_feature_dim + timestamp_feature_dim
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(combined_feature_dim, num_classes)

    def forward(self, input_ids, attention_mask, sequences, sender_names, receiver_names, timestamps):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = bert_outputs.pooler_output
        all_relationship_strengths = self.relationship_lstm(sequences)

        batch_size = input_ids.size(0)
        sender_receiver_strengths = []
        default_strength = torch.tensor(0.0, device=input_ids.device, dtype=torch.float)
        edge_indices = []
        for i in range(batch_size):
            edge_idx = get_relationship_index(sender_names[i], receiver_names[i])
            strength = all_relationship_strengths[i, edge_idx] if edge_idx is not None else default_strength
            sender_receiver_strengths.append(strength)
            edge_indices.append(edge_idx if edge_idx is not None else -1)

        sender_receiver_strengths_tensor = torch.stack(sender_receiver_strengths).unsqueeze(1)
        scores = sequences[:, 0, :NUM_PLAYERS]
        combined_features = torch.cat([text_embedding, scores, sender_receiver_strengths_tensor, timestamps], dim=1)
        dropped_features = self.dropout(combined_features)
        logits = self.classifier(dropped_features)

        return logits, all_relationship_strengths, torch.tensor(edge_indices, device=input_ids.device)


# Helper Functions
def get_relationship_index(sender_name, receiver_name):
    pair = tuple(sorted((sender_name, receiver_name)))
    return PAIR_TO_EDGE_IDX.get(pair, None)


def extract_direct_scores(game_data, message_index):
    scores = {}
    try:
        sender_score_str = game_data["game_score"][message_index]
        delta_score_str = game_data["game_score_delta"][message_index]
        sender = game_data["speakers"][message_index]
        receiver = game_data["receivers"][message_index]
        sender_score = int(sender_score_str)
        delta_score = int(delta_score_str)
        receiver_score = sender_score - delta_score
        if sender in PLAYER_TO_IDX: scores[sender] = sender_score
        if receiver in PLAYER_TO_IDX: scores[receiver] = receiver_score
    except (IndexError, ValueError, KeyError, TypeError):
        pass
    return scores


def load_and_preprocess(file_path):
    processed_data = []
    game_scores = {}
    game_max_indices = {}
    print(f"Processing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(file_path)}"):
            try:
                game_dialog = json.loads(line.strip())
                num_messages = len(game_dialog.get("messages", []))
                if num_messages == 0: continue
                required_keys = ["sender_labels", "receiver_labels", "speakers", "receivers", "absolute_message_index",
                                 "game_score", "game_score_delta", "seasons", "years"]
                if not all(k in game_dialog for k in required_keys): continue

                game_id = game_dialog.get("game_id", "unknown")
                if game_id not in game_scores:
                    game_scores[game_id] = {}
                if game_id not in game_max_indices:
                    game_max_indices[game_id] = 0

                # Track scores for all players
                for i in range(num_messages):
                    abs_idx = game_dialog["absolute_message_index"][i]
                    direct_scores = extract_direct_scores(game_dialog, i)
                    if abs_idx not in game_scores[game_id]:
                        game_scores[game_id][abs_idx] = {p: SCORE_IMPUTATION_VALUE for p in ALL_PLAYERS}
                    for player, score in direct_scores.items():
                        game_scores[game_id][abs_idx][player] = score
                    game_max_indices[game_id] = max(game_max_indices[game_id], abs_idx)

                # Fill in missing scores with last known value or imputation
                sorted_indices = sorted(game_scores[game_id].keys())
                for i, idx in enumerate(sorted_indices):
                    current_scores = game_scores[game_id][idx]
                    if i > 0:
                        prev_scores = game_scores[game_id][sorted_indices[i - 1]]
                        for p in ALL_PLAYERS:
                            if current_scores[p] == SCORE_IMPUTATION_VALUE:
                                current_scores[p] = prev_scores[p]

                # Create instances
                for i in range(num_messages):
                    required_lists = ["messages", "sender_labels", "receiver_labels", "speakers", "receivers",
                                      "absolute_message_index", "relative_message_index", "game_score",
                                      "game_score_delta", "seasons", "years"]
                    if not all(len(game_dialog.get(key, [])) > i for key in required_lists): continue
                    message_text = game_dialog["messages"][i]
                    sender_label = 0 if game_dialog["sender_labels"][i] else 1
                    receiver_label_raw = game_dialog["receiver_labels"][i]
                    receiver_label = -1 if receiver_label_raw == "NOANNOTATION" else (0 if receiver_label_raw else 1)
                    sender = game_dialog["speakers"][i]
                    receiver = game_dialog["receivers"][i]
                    abs_msg_index = game_dialog["absolute_message_index"][i]
                    season = game_dialog["seasons"][i]
                    year = game_dialog["years"][i]
                    if sender not in PLAYER_TO_IDX or receiver not in PLAYER_TO_IDX: continue
                    instance = {
                        "game_id": game_id,
                        "message_text": message_text,
                        "sender": sender,
                        "receiver": receiver,
                        "sender_label": sender_label,
                        "receiver_label": receiver_label,
                        "all_scores": game_scores[game_id][abs_idx],
                        "absolute_message_index": abs_msg_index,
                        "season": season,
                        "year": year,
                    }
                    processed_data.append(instance)
            except Exception as e:
                print(f"Error processing line: {e}")
    print(f"Finished {file_path}. Found {len(processed_data)} valid instances.")
    return processed_data, game_max_indices


class DiplomacyDataset(torch.utils.data.Dataset):
    def __init__(self, data_flat, tokenizer, max_len, game_max_indices, score_imputation_value=SCORE_IMPUTATION_VALUE):
        self.data = data_flat
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.game_max_indices = game_max_indices
        self.score_imputation_value = float(score_imputation_value)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        message = item['message_text']
        sender_label = item['sender_label']
        receiver_label = item['receiver_label']
        sender_name = item['sender']
        receiver_name = item['receiver']
        abs_msg_index = item['absolute_message_index']
        game_id = item['game_id']
        season = item['season']
        year = item['year']

        encoding = self.tokenizer.encode_plus(
            message, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )

        score_vector = np.array([item['all_scores'][p] for p in ALL_PLAYERS], dtype=np.float32)
        timestamp = abs_msg_index / (self.game_max_indices.get(game_id, 1) + 1)
        season_idx = SEASON_TO_IDX.get(season, 0)
        year_normalized = (int(year) - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)

        sequence = np.concatenate([score_vector, [season_idx, year_normalized, timestamp]], axis=0).astype(np.float32)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sender_label': torch.tensor(sender_label, dtype=torch.long),
            'receiver_label': torch.tensor(receiver_label, dtype=torch.long),
            'sequence': torch.tensor(sequence, dtype=torch.float),
            'sender_name': sender_name,
            'receiver_name': receiver_name,
            'timestamp': torch.tensor(timestamp, dtype=torch.float)
        }


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    sender_labels = torch.stack([item['sender_label'] for item in batch])
    receiver_labels = torch.stack([item['receiver_label'] for item in batch])
    sequences = torch.stack([item['sequence'] for item in batch]).unsqueeze(1)
    timestamps = torch.stack([item['timestamp'] for item in batch]).unsqueeze(1)
    sender_names = [item['sender_name'] for item in batch]
    receiver_names = [item['receiver_name'] for item in batch]
    valid_receiver_mask = (receiver_labels != -1)
    return {
        'input_ids': input_ids, 'attention_mask': attention_mask, 'sender_labels': sender_labels,
        'receiver_labels': receiver_labels, 'sequences': sequences, 'sender_names': sender_names,
        'receiver_names': receiver_names, 'timestamps': timestamps, 'valid_receiver_mask': valid_receiver_mask
    }


def run_inference(model, data_loader, device, task='sender'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Inferencing ({task})"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sequences = batch['sequences'].to(device)
            sender_names = batch['sender_names']
            receiver_names = batch['receiver_names']
            timestamps = batch['timestamps'].to(device)

            if task == 'sender':
                labels = batch['sender_labels'].to(device)
                valid_mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                labels = batch['receiver_labels'].to(device)
                valid_mask = batch['valid_receiver_mask'].to(device)

            logits, _, _ = model(input_ids, attention_mask, sequences, sender_names, receiver_names, timestamps)

            if task == 'receiver':
                logits = logits[valid_mask]
                labels = labels[valid_mask]

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    # Filter out any -1 labels (no annotation)
    valid_indices = all_labels != -1
    all_preds = all_preds[valid_indices]
    all_labels = all_labels[valid_indices]

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    lie_f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)

    return accuracy, macro_f1, lie_f1


def main():
    # Configuration
    project_dir = os.getcwd()
    data_dir = os.path.join("../dataset")
    output_dir = os.path.join(project_dir, "saved_models")
    test_path = os.path.join(data_dir, "test.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load test data
    test_data_flat, test_max_indices = load_and_preprocess(test_path)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    test_dataset = DiplomacyDataset(test_data_flat, tokenizer, MAX_LEN, test_max_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Models to evaluate
    tasks = ['sender', 'receiver']
    model_types = ['macro', 'lie']

    for task in tasks:
        print(f"\n{'=' * 40}")
        print(f"Task: {task.upper()}")
        print(f"{'=' * 40}")

        for model_type in model_types:
            model_path = os.path.join(output_dir, f'diplomacy_{model_type}_model_{task}_best.bin')

            if os.path.exists(model_path):
                model = DiplomacyGraphModel()
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)

                print(f"\nEvaluating {task} model optimized for {model_type} F1...")
                accuracy, macro_f1, lie_f1 = run_inference(model, test_loader, device, task)

                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"Test Macro F1: {macro_f1:.4f}")
                print(f"Test Lie F1: {lie_f1:.4f}")
            else:
                print(f"Model file not found: {model_path}")


if __name__ == "__main__":
    main()
