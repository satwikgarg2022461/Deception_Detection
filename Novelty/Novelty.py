import os
import json
import random
import re
import collections
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt

# --- Set Seed for Reproducibility ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Configuration ---
project_dir = os.getcwd()
data_dir = os.path.join("../dataset")
output_dir = os.path.join(project_dir, "saved_models")
os.makedirs(output_dir, exist_ok=True)

test_path = os.path.join(data_dir, "test.jsonl")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")

BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-5
SCORE_IMPUTATION_VALUE = 3.0
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Constants ---
ALL_PLAYERS = ["england", "france", "germany", "italy", "austria", "russia", "turkey"]
PLAYER_TO_IDX = {name: i for i, name in enumerate(ALL_PLAYERS)}
NUM_PLAYERS = len(ALL_PLAYERS)
PLAYER_PAIRS = sorted([tuple(sorted(pair)) for pair in itertools.combinations(ALL_PLAYERS, 2)])
PAIR_TO_EDGE_IDX = {pair: i for i, pair in enumerate(PLAYER_PAIRS)}
NUM_EDGES = len(PLAYER_PAIRS)
SEASONS = ["fall", "winter", "spring"]
SEASON_TO_IDX = {season: i for i, season in enumerate(SEASONS)}
MIN_YEAR = 1901
MAX_YEAR = 1909


# --- Helper Functions ---
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
    game_scores = {}  # game_id -> absolute_message_index -> player -> score
    game_max_indices = {}  # game_id -> max absolute_message_index
    print(f"Processing {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

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


class RelationshipLSTM(nn.Module):
    def __init__(self, input_dim=NUM_PLAYERS + 3, hidden_dim=64, output_dim=NUM_EDGES, num_layers=1):
        super().__init__()
        self.input_dim = input_dim  # All player scores (7) + season + year + timestamp
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        # sequence: [batch_size, seq_len, input_dim]
        lstm_out, (hn, cn) = self.lstm(sequence)
        # Take the last output of the LSTM
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out  # No sigmoid, raw output


def get_relationship_index(sender_name, receiver_name):
    pair = tuple(sorted((sender_name, receiver_name)))
    return PAIR_TO_EDGE_IDX.get(pair, None)


class DiplomacyDataset(Dataset):
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

        # All player scores
        score_vector = np.array([item['all_scores'][p] for p in ALL_PLAYERS], dtype=np.float32)
        timestamp = abs_msg_index / (self.game_max_indices.get(game_id, 1) + 1)  # Normalize
        season_idx = SEASON_TO_IDX.get(season, 0)  # Default to 0 if season not found
        year_normalized = (int(year) - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)  # Normalize year to [0, 1]

        # Create sequence for LSTM: [all scores (7), season, year, timestamp]
        sequence = np.concatenate([score_vector, [season_idx, year_normalized, timestamp]], axis=0).astype(np.float32)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sender_label': torch.tensor(sender_label, dtype=torch.long),
            'receiver_label': torch.tensor(receiver_label, dtype=torch.long),
            'sequence': torch.tensor(sequence, dtype=torch.float),  # Sequence for LSTM
            'sender_name': sender_name,
            'receiver_name': receiver_name,
            'timestamp': torch.tensor(timestamp, dtype=torch.float)
        }


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    sender_labels = torch.stack([item['sender_label'] for item in batch])
    receiver_labels = torch.stack([item['receiver_label'] for item in batch])
    sequences = torch.stack([item['sequence'] for item in batch]).unsqueeze(1)  # [batch_size, seq_len=1, input_dim]
    timestamps = torch.stack([item['timestamp'] for item in batch]).unsqueeze(1)
    sender_names = [item['sender_name'] for item in batch]
    receiver_names = [item['receiver_name'] for item in batch]
    valid_receiver_mask = (receiver_labels != -1)
    return {
        'input_ids': input_ids, 'attention_mask': attention_mask, 'sender_labels': sender_labels,
        'receiver_labels': receiver_labels, 'sequences': sequences, 'sender_names': sender_names,
        'receiver_names': receiver_names, 'timestamps': timestamps, 'valid_receiver_mask': valid_receiver_mask
    }


class DiplomacyGraphModel(nn.Module):
    def __init__(self, bert_model_name=BERT_MODEL_NAME, lstm_hidden_dim=64, dropout_rate=0.1, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.relationship_lstm = RelationshipLSTM(input_dim=NUM_PLAYERS + 3,
                                                  hidden_dim=lstm_hidden_dim)  # All scores + season + year + timestamp
        bert_output_dim = self.bert.config.hidden_size
        scores_dim = NUM_PLAYERS  # All player scores
        relationship_feature_dim = 1
        timestamp_feature_dim = 1
        combined_feature_dim = bert_output_dim + scores_dim + relationship_feature_dim + timestamp_feature_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(combined_feature_dim, num_classes)

    def forward(self, input_ids, attention_mask, sequences, sender_names, receiver_names, timestamps):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = bert_outputs.pooler_output
        all_relationship_strengths = self.relationship_lstm(sequences)  # [batch_size, NUM_EDGES]

        batch_size = input_ids.size(0)
        sender_receiver_strengths = []
        default_strength = torch.tensor(0.0, device=input_ids.device,
                                        dtype=torch.float)  # Changed to 0.0 since no sigmoid
        edge_indices = []
        for i in range(batch_size):
            edge_idx = get_relationship_index(sender_names[i], receiver_names[i])
            strength = all_relationship_strengths[i, edge_idx] if edge_idx is not None else default_strength
            sender_receiver_strengths.append(strength)
            edge_indices.append(edge_idx if edge_idx is not None else -1)

        sender_receiver_strengths_tensor = torch.stack(sender_receiver_strengths).unsqueeze(1)
        scores = sequences[:, 0,
                 :NUM_PLAYERS]  # Extract all scores from the sequence [batch_size, 1, NUM_PLAYERS] -> [batch_size, NUM_PLAYERS]
        combined_features = torch.cat([text_embedding, scores, sender_receiver_strengths_tensor, timestamps], dim=1)
        dropped_features = self.dropout(combined_features)
        logits = self.classifier(dropped_features)

        return logits, all_relationship_strengths, torch.tensor(edge_indices, device=input_ids.device)


def evaluate_model(model, data_loader, loss_fn_sender, loss_fn_receiver, device, task='sender'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating ({task})", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sequences = batch['sequences'].to(device)
            sender_names = batch['sender_names']
            receiver_names = batch['receiver_names']
            timestamps = batch['timestamps'].to(device)
            sender_labels = batch['sender_labels'].to(device)
            receiver_labels = batch['receiver_labels'].to(device)
            valid_receiver_mask = batch['valid_receiver_mask'].to(device)

            logits, _, _ = model(input_ids, attention_mask, sequences, sender_names, receiver_names, timestamps)

            if task == 'sender':
                loss = loss_fn_sender(logits, sender_labels)
                labels = sender_labels
                preds = torch.argmax(logits, dim=1)
            elif task == 'receiver':
                valid_logits = logits[valid_receiver_mask]
                valid_labels = receiver_labels[valid_receiver_mask]
                if valid_labels.numel() > 0:
                    loss = loss_fn_receiver(valid_logits, valid_labels)
                    preds = torch.argmax(valid_logits, dim=1)
                    labels = valid_labels
                else:
                    loss = torch.tensor(0.0).to(device)
                    preds = torch.tensor([], dtype=torch.long).to(device)
                    labels = torch.tensor([], dtype=torch.long).to(device)

            if labels.numel() > 0:
                total_loss += loss.item() * (labels.numel() / batch['sender_labels'].size(0))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    if all_preds.size > 0 and all_labels.size > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        lie_f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
        return avg_loss, macro_f1, lie_f1, accuracy
    return avg_loss, 0.0, 0.0, 0.0


def train_and_evaluate(task_name, train_loader, val_loader, test_loader, device):
    print(f"\n{'=' * 15} Training Model for Task: {task_name.upper()} {'=' * 15}")

    model = DiplomacyGraphModel().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    sender_labels = torch.tensor([item['sender_label'] for item in train_loader.dataset.data])
    receiver_labels = torch.tensor([item['receiver_label'] for item in train_loader.dataset.data])

    sender_valid_labels = sender_labels
    sender_class_counts = torch.bincount(sender_valid_labels)
    sender_class_weights = torch.tensor([1.0, 30.0], dtype=torch.float)

    receiver_valid_labels = receiver_labels[receiver_labels != -1]
    receiver_class_counts = torch.bincount(receiver_valid_labels)
    receiver_class_weights = torch.tensor([1.0, 50.0], dtype=torch.float)

    sender_loss_fn = nn.CrossEntropyLoss(weight=sender_class_weights.to(device))
    receiver_loss_fn = nn.CrossEntropyLoss(weight=receiver_class_weights.to(device), ignore_index=-1)

    if task_name == 'sender':
        print(f"Sender Class counts (0=truth, 1=lie): {sender_class_counts.tolist()}")
        print(f"Sender Class weights: {sender_class_weights.tolist()}")
    else:
        print(f"Receiver Class counts (0=truth, 1=lie): {receiver_class_counts.tolist()}")
        print(f"Receiver Class weights: {receiver_class_weights.tolist()}")

    best_val_lie_f1 = 0.0
    best_val_macro_f1 = 0.0
    best_model_macro_path = os.path.join(output_dir, f'diplomacy_macro_model_{task_name}_best.bin')
    best_model_lie_path = os.path.join(output_dir, f'diplomacy_lie_model_{task_name}_best.bin')

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training ({task_name})", leave=False)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sequences = batch['sequences'].to(device)
            sender_names = batch['sender_names']
            receiver_names = batch['receiver_names']
            timestamps = batch['timestamps'].to(device)
            sender_labels = batch['sender_labels'].to(device)
            receiver_labels = batch['receiver_labels'].to(device)
            valid_receiver_mask = batch['valid_receiver_mask'].to(device)

            model.zero_grad()
            logits, all_relationship_strengths, edge_indices = model(
                input_ids, attention_mask, sequences, sender_names, receiver_names, timestamps
            )

            if task_name == 'sender':
                labels = sender_labels
                loss_full = sender_loss_fn(logits, labels)
            else:
                valid_logits = logits[valid_receiver_mask]
                valid_labels = receiver_labels[valid_receiver_mask]
                loss_full = receiver_loss_fn(valid_logits, valid_labels) if valid_labels.numel() > 0 else torch.tensor(
                    0.0, device=device)
                labels = receiver_labels

            batch_size = input_ids.size(0)
            edge_mask = torch.ones_like(all_relationship_strengths)
            for i, idx in enumerate(edge_indices):
                if idx != -1:
                    edge_mask[i, :idx] = 0
                    edge_mask[i, idx + 1:] = 0
            relationship_loss = ((all_relationship_strengths * edge_mask) ** 2).mean()
            loss = loss_full + 0.01 * relationship_loss

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss_full.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f"{loss_full.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        valid_indices = all_labels != -1
        all_preds = all_preds[valid_indices]
        all_labels = all_labels[valid_indices]
        train_accuracy = accuracy_score(all_labels, all_preds)  # Calculate training accuracy
        train_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        train_lie_f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)

        print(f"Average Training Loss ({task_name}): {avg_train_loss:.4f}")
        print(f"Training Accuracy ({task_name}): {train_accuracy:.4f}")
        print(f"Training Macro F1 ({task_name}): {train_macro_f1:.4f}")
        print(f"Training Lie F1 ({task_name}): {train_lie_f1:.4f}")

        val_loss, val_macro_f1, val_lie_f1, val_accuracy = evaluate_model(model, val_loader, sender_loss_fn,
                                                                          receiver_loss_fn, device, task=task_name)
        print(f"Validation Loss ({task_name}): {val_loss:.4f}")
        print(f"Validation Accuracy ({task_name}): {val_accuracy:.4f}")
        print(f"Validation Macro F1 ({task_name}): {val_macro_f1:.4f}")
        print(f"Validation Lie F1 ({task_name}): {val_lie_f1:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)  # Store actual accuracy instead of F1
        val_accuracies.append(val_accuracy)  # Store validation accuracy

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_model_macro_path)
            print(f"New best model saved with Macro F1: {best_val_macro_f1:.4f}")
            print(f"\n--- Final Test Evaluation Macro ({task_name.upper()}) ---")
            model.load_state_dict(torch.load(best_model_macro_path))
            test_loss, test_macro_f1, test_lie_f1, test_accuracy = evaluate_model(model, test_loader, sender_loss_fn,
                                                                                  receiver_loss_fn, device,
                                                                                  task=task_name)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Macro F1: {test_macro_f1:.4f}")
            print(f"Test Lie F1: {test_lie_f1:.4f}")

        if val_lie_f1 > best_val_lie_f1:
            best_val_lie_f1 = val_lie_f1
            torch.save(model.state_dict(), best_model_lie_path)
            print(f"New best model saved with Lie F1: {best_val_lie_f1:.4f}")
            print(f"\n--- Final Test Evaluation Lie ({task_name.upper()}) ---")
            model.load_state_dict(torch.load(best_model_lie_path))
            test_loss, test_macro_f1, test_lie_f1, test_accuracy = evaluate_model(model, test_loader, sender_loss_fn,
                                                                                  receiver_loss_fn, device,
                                                                                  task=task_name)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Macro F1: {test_macro_f1:.4f}")
            print(f"Test Lie F1: {test_lie_f1:.4f}")

    print(f"\n--- Final Test Evaluation Macro ({task_name.upper()}) ---")
    model.load_state_dict(torch.load(best_model_macro_path))
    test_loss, test_macro_f1, test_lie_f1, test_accuracy = evaluate_model(model, test_loader, sender_loss_fn,
                                                                          receiver_loss_fn, device, task=task_name)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Lie F1: {test_lie_f1:.4f}")

    print(f"\n--- Final Test Evaluation Lie ({task_name.upper()}) ---")
    model.load_state_dict(torch.load(best_model_lie_path))
    test_loss, test_macro_f1, test_lie_f1, test_accuracy = evaluate_model(model, test_loader, sender_loss_fn,
                                                                          receiver_loss_fn, device, task=task_name)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Lie F1: {test_lie_f1:.4f}")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, task_name)


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, task_name):
    """
    Plot training and validation metrics
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title(f'{task_name.capitalize()} Task - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title(f'{task_name.capitalize()} Task - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{task_name}_metrics.png"))
    print(f"Metrics plot saved to {task_name}_metrics.png")
    plt.close()


# --- Main Execution ---
train_data_flat, train_max_indices = load_and_preprocess(train_path)
val_data_flat, val_max_indices = load_and_preprocess(val_path)
test_data_flat, test_max_indices = load_and_preprocess(test_path)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
train_dataset = DiplomacyDataset(train_data_flat, tokenizer, MAX_LEN, train_max_indices)
val_dataset = DiplomacyDataset(val_data_flat, tokenizer, MAX_LEN, val_max_indices)
test_dataset = DiplomacyDataset(test_data_flat, tokenizer, MAX_LEN, test_max_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

train_and_evaluate('sender', train_loader, val_loader, test_loader, device)
train_and_evaluate('receiver', train_loader, val_loader, test_loader, device)

print("\n--- All Tasks Finished ---")
