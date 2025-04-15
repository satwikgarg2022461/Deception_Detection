





# Still in development







import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import jsonlines
import os
import argparse
from tqdm import tqdm  # Use this import only
from collections import Counter
from transformers import BertTokenizer, BertModel
import random
import pickle

# Set paths
project_dir = ""
data_dir = os.path.join(project_dir, "dataset")
models_dir = os.path.join(project_dir, "models")
test_path = os.path.join(data_dir, "test.jsonl")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(1994)
np.random.seed(1994)
random.seed(1994)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()  # Freeze BERT weights

def tokenize_with_bert(text, max_length=20):
    """Tokenize text using BERT's WordPiece tokenizer"""
    encoding = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

def encode_message_batch(messages, attention_masks):
    """Encode a batch of messages using BERT"""
    with torch.no_grad():
        inputs = {'input_ids': messages.to(device), 'attention_mask': attention_masks.to(device)}
        outputs = bert_model(**inputs)
        # Use the [CLS] token's embedding (pooled output) from the pooler layer
        return outputs.pooler_output  # Shape: [batch_size, 768]

class ConversationDataset(Dataset):
    """Dataset for conversation-level processing with BERT"""

    def __init__(self, data, max_length=20, max_conv_length=10, use_power=False, task="sender"):
        self.data = data
        self.max_length = max_length
        self.max_conv_length = max_conv_length
        self.use_power = use_power
        self.task = task
        self.processed_data = []
        self.process_conversations()

    def process_conversations(self):
        """Process conversations to maintain message context"""
        for conversation in self.data:
            conv_messages = []
            conv_masks = []
            conv_powers = []
            conv_labels = []

            for i, message in enumerate(conversation['messages']):
                if self.task.lower() == "sender":
                    input_ids, attention_mask = tokenize_with_bert(message, self.max_length)
                    conv_messages.append(input_ids)
                    conv_masks.append(attention_mask)
                    if self.use_power:
                        conv_powers.append([
                            1 if int(conversation['game_score_delta'][i]) > 4 else 0,
                            1 if int(conversation['game_score_delta'][i]) < -4 else 0
                        ])
                    label = 0 if conversation['sender_labels'][i] else 1
                    conv_labels.append(label)
                else:  # receiver task
                    if conversation['receiver_labels'][i] not in [True, False]:
                        continue
                    input_ids, attention_mask = tokenize_with_bert(message, self.max_length)
                    conv_messages.append(input_ids)
                    conv_masks.append(attention_mask)
                    if self.use_power:
                        conv_powers.append([
                            1 if int(conversation['game_score_delta'][i]) > 4 else 0,
                            1 if int(conversation['game_score_delta'][i]) < -4 else 0
                        ])
                    label = 0 if conversation['receiver_labels'][i] else 1
                    conv_labels.append(label)

            if len(conv_messages) == 0:
                continue

            if len(conv_messages) > self.max_conv_length:
                conv_messages = conv_messages[-self.max_conv_length:]
                conv_masks = conv_masks[-self.max_conv_length:]
                conv_labels = conv_labels[-self.max_conv_length:]
                if self.use_power:
                    conv_powers = conv_powers[-self.max_conv_length:]

            if self.use_power:
                self.processed_data.append({
                    'messages': conv_messages,
                    'masks': conv_masks,
                    'powers': conv_powers,
                    'labels': conv_labels
                })
            else:
                self.processed_data.append({
                    'messages': conv_messages,
                    'masks': conv_masks,
                    'labels': conv_labels
                })

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        data = self.processed_data[idx]
        messages = torch.stack(data['messages'])
        masks = torch.stack(data['masks'])
        labels = torch.tensor(data['labels'], dtype=torch.long)
        if self.use_power:
            powers = torch.tensor(data['powers'], dtype=torch.float)
            return messages, masks, powers, labels
        return messages, masks, labels

def collate_batch(batch):
    """Custom collate function for BERT-based data"""
    has_power = len(batch[0]) == 4
    if has_power:
        messages, masks, powers, labels = zip(*batch)
        powers = torch.cat(powers)
    else:
        messages, masks, labels = zip(*batch)

    all_messages = torch.cat(messages)
    all_masks = torch.cat(masks)
    conv_labels = [conv_label[-1] for conv_label in labels]
    all_labels = torch.tensor(conv_labels, dtype=torch.long)

    conv_boundaries = [0]
    total_msgs = 0
    for msgs in messages:
        total_msgs += len(msgs)
        conv_boundaries.append(total_msgs)

    if has_power:
        return all_messages, all_masks, conv_boundaries, powers, all_labels
    return all_messages, all_masks, conv_boundaries, all_labels

class ContextLSTMModel(nn.Module):
    """Hierarchical LSTM model with BERT message encoder"""

    def __init__(self, conv_hidden_dim, output_dim, use_power=False, dropout=0.1):
        super(ContextLSTMModel, self).__init__()
        self.bert_dim = 768  # BERT pooler output size
        self.conv_lstm = nn.LSTM(self.bert_dim, conv_hidden_dim, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.use_power = use_power
        self.fc = nn.Linear(conv_hidden_dim + 2 if use_power else conv_hidden_dim, output_dim)

    def forward(self, messages, masks, conv_boundaries, power=None):
        # messages shape: [total_messages, max_length]
        # Encode all messages with BERT
        message_repr = encode_message_batch(messages, masks)  # [total_messages, 768]
        message_repr = self.dropout(message_repr)

        # Group messages by conversation
        conversation_inputs = []
        for i in range(len(conv_boundaries) - 1):
            start, end = conv_boundaries[i], conv_boundaries[i + 1]
            conversation_inputs.append(message_repr[start:end])

        padded_convs = pad_sequence(conversation_inputs, batch_first=True)
        _, (conv_hidden, _) = self.conv_lstm(padded_convs)
        conv_hidden = conv_hidden.squeeze(0)  # [batch_size, conv_hidden_dim]
        conv_hidden = self.dropout(conv_hidden)

        if self.use_power and power is not None:
            batch_powers = torch.stack([power[end - 1] for i in range(len(conv_boundaries) - 1)
                                       for start, end in [(conv_boundaries[i], conv_boundaries[i + 1])]])
            combined = torch.cat((conv_hidden, batch_powers), dim=1)
            output = self.fc(combined)
        else:
            output = self.fc(conv_hidden)
        return output

def train_epoch(model, data_loader, optimizer, criterion, device, use_power=False):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        if use_power:
            messages, masks, conv_boundaries, powers, labels = batch
            # Move only tensors to device
            messages = messages.to(device)
            masks = masks.to(device)
            powers = powers.to(device)
            labels = labels.to(device)
            # conv_boundaries remains a Python list
            optimizer.zero_grad()
            predictions = model(messages, masks, conv_boundaries, powers)
        else:
            messages, masks, conv_boundaries, labels = batch
            # Move only tensors to device
            messages = messages.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            # conv_boundaries remains a Python list
            optimizer.zero_grad()
            predictions = model(messages, masks, conv_boundaries)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(predictions, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

def evaluate(model, data_loader, criterion, device, use_power=False):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds, all_labels = [], []
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            if use_power:
                messages, masks, conv_boundaries, powers, labels = batch
                messages = messages.to(device)
                masks = masks.to(device)
                powers = powers.to(device)
                labels = labels.to(device)
                predictions = model(messages, masks, conv_boundaries, powers)
            else:
                messages, masks, conv_boundaries, labels = batch
                messages = messages.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                predictions = model(messages, masks, conv_boundaries)
            loss = criterion(predictions, labels)
            preds = torch.argmax(predictions, dim=1)
            acc = torch.sum(preds == labels).float() / len(labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader), all_preds, all_labels

def run_contextlstm_model(task="sender", use_power=False, save_model=True):
    print(f"Loading data for {task.upper()} task with power={use_power}")
    with jsonlines.open(train_path, 'r') as reader:
        train_data = list(reader)
    with jsonlines.open(val_path, 'r') as reader:
        val_data = list(reader)
    with jsonlines.open(test_path, 'r') as reader:
        test_data = list(reader)

    train_dataset = ConversationDataset(train_data, max_length=20, max_conv_length=10, use_power=use_power, task=task)
    val_dataset = ConversationDataset(val_data, max_length=20, max_conv_length=10, use_power=use_power, task=task)
    test_dataset = ConversationDataset(test_data, max_length=20, max_conv_length=10, use_power=use_power, task=task)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

    conv_hidden_dim = 200
    output_dim = 2
    dropout = 0.4 if task.lower() == "sender" else 0.1
    model = ContextLSTMModel(conv_hidden_dim=conv_hidden_dim, output_dim=output_dim, use_power=use_power, dropout=dropout).to(device)

    positive_weight = 15.0 if task.lower() == "sender" else 50.0
    weight = torch.tensor([1.0, positive_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    n_epochs = 15
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None
    best_val_metrics = None

    print(f"Starting training for {n_epochs} epochs...")
    for epoch in tqdm(range(n_epochs), desc="Training epochs"):  # Correct usage of tqdm
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, use_power)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device, use_power)
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
        val_binary_f1 = f1_score(val_labels, val_preds, average='binary', pos_label=1)
        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}, Val Binary F1: {val_binary_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model = model.state_dict().copy()
            best_val_metrics = {'loss': val_loss, 'acc': val_acc, 'macro_f1': val_macro_f1, 'binary_f1': val_binary_f1, 'epoch': epoch + 1}
            print("New best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device, use_power)

    print('test pred',test_preds)
    print('test labels',test_labels)
    metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'macro_f1': f1_score(test_labels, test_preds, average='macro'),
        'binary_f1': f1_score(test_labels, test_preds, average='binary', pos_label=1),
        'precision': precision_score(test_labels, test_preds, pos_label=1, zero_division=0),
        'recall': recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    }

    if save_model and best_model is not None:
        power_suffix = "_with_power" if use_power else "_without_power"
        model_filename = f"contextlstm_{task}{power_suffix}.pt"
        model_path = os.path.join(models_dir, model_filename)
        torch.save({
            'model_state_dict': best_model,
            'conv_hidden_dim': conv_hidden_dim,
            'val_metrics': best_val_metrics,
            'config': {'task': task, 'use_power': use_power, 'dropout': dropout}
        }, model_path)
        print(f"\nBest model saved to {model_path}")
        print(f"Best validation metrics: Loss={best_val_metrics['loss']:.4f}, Macro F1={best_val_metrics['macro_f1']:.4f} (Epoch {best_val_metrics['epoch']})")

    print(f"\n=== ContextLSTM Results for {task.upper()} Task ===")
    print(f"Power features: {'Yes' if use_power else 'No'}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Binary/Lie F1: {metrics['binary_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, digits=4, target_names=['Truthful', 'Deceptive']))
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ContextLSTM model for deception detection')
    parser.add_argument('--task', choices=['sender', 'receiver'], default='sender')
    parser.add_argument('--power', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    metrics = run_contextlstm_model(task=args.task, use_power=args.power, save_model=not args.no_save)

    # Compare with paper results
    paper_results = {
        'sender': {
            'with_power': {'macro_f1': 0.572, 'binary_f1': 0.27},
            'without_power': {'macro_f1': 0.558, 'binary_f1': 0.192},
        },
        'receiver': {
            'with_power': {'macro_f1': 0.533, 'binary_f1': 0.13},
            'without_power': {'macro_f1': 0.543, 'binary_f1': 0.15},
        }
    }

    power_key = 'with_power' if args.power else 'without_power'
    paper_macro_f1 = paper_results[args.task][power_key]['macro_f1']
    paper_binary_f1 = paper_results[args.task][power_key]['binary_f1']

    print("\n=== Comparison with Paper Results ===")
    print(f"Our Macro F1: {metrics['macro_f1']:.4f}   Paper: {paper_macro_f1:.4f}")
    print(f"Our Lie F1: {metrics['binary_f1']:.4f}   Paper: {paper_binary_f1:.4f}")