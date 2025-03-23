import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import jsonlines
import os
import re
import argparse
import tqdm
from collections import Counter, defaultdict
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

# Set paths
project_dir = "d:\\NLP\\Deception_Detection"
data_dir = os.path.join(project_dir,  "dataset")
test_path = os.path.join(data_dir, "test.jsonl")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")

# Create models directory if it doesn't exist
models_dir = os.path.join(project_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert conversations into single messages for data processing
def aggregate(dataset):
    messages = []
    rec = []
    send = []
    power = []
    for dialogs in dataset:
        messages.extend(dialogs['messages'])
        rec.extend(dialogs['receiver_labels'])
        send.extend(dialogs['sender_labels'])
        # Add power data
        power.extend(dialogs['game_score_delta'])
    merged = []
    for i, item in enumerate(messages):
        merged.append({
            'message': item, 
            'sender_annotation': send[i], 
            'receiver_annotation': rec[i],
            'score_delta': int(power[i])
        })
    return merged

class ConversationBERTDataset(Dataset):
    """Dataset for conversation-level processing with BERT"""
    def __init__(self, data, tokenizer, max_length=128, max_conv_length=10, use_power=False, task="sender"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_conv_length = max_conv_length
        self.use_power = use_power
        self.task = task
        
        # Process conversations
        self.processed_data = []
        self.process_conversations()
    
    def process_conversations(self):
        """Process conversations to maintain message context"""
        for conversation in self.data:
            conv_encodings = []
            conv_powers = []
            conv_labels = []
            
            for i, message in enumerate(conversation['messages']):
                # Process message text with BERT tokenizer
                encoding = self.tokenizer(
                    message,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Extract input_ids and attention_mask
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                
                # Store encoding
                conv_encodings.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
                
                # Process power features
                if self.use_power:
                    conv_powers.append([
                        1 if int(conversation['game_score_delta'][i]) > 4 else 0,  # Sender much stronger
                        1 if int(conversation['game_score_delta'][i]) < -4 else 0   # Receiver much stronger
                    ])
                
                # Get label based on task
                if self.task.lower() == "sender":
                    label = 0 if conversation['sender_labels'][i] else 1  # 0 = truthful, 1 = deceptive
                    conv_labels.append(label)
                else:  # receiver task
                    # Skip if no receiver annotation
                    if conversation['receiver_labels'][i] not in [True, False]:
                        continue
                    label = 0 if conversation['receiver_labels'][i] else 1  # 0 = truthful, 1 = deceptive
                    conv_labels.append(label)
            
            # Skip if no valid messages in this conversation
            if len(conv_encodings) == 0:
                continue
                
            # Truncate if too many messages
            if len(conv_encodings) > self.max_conv_length:
                conv_encodings = conv_encodings[-self.max_conv_length:]
                conv_labels = conv_labels[-self.max_conv_length:]
                if self.use_power:
                    conv_powers = conv_powers[-self.max_conv_length:]
            
            # Add to processed data
            entry = {
                'encodings': conv_encodings,
                'labels': conv_labels
            }
            if self.use_power:
                entry['powers'] = conv_powers
                
            self.processed_data.append(entry)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        data = self.processed_data[idx]
        
        # Extract encodings
        input_ids = torch.stack([encoding['input_ids'] for encoding in data['encodings']])
        attention_mask = torch.stack([encoding['attention_mask'] for encoding in data['encodings']])
        labels = torch.tensor(data['labels'], dtype=torch.long)
        
        if self.use_power:
            powers = torch.tensor(data['powers'], dtype=torch.float)
            return input_ids, attention_mask, powers, labels
        else:
            return input_ids, attention_mask, labels

def collate_bert_batch(batch):
    """Custom collate function for BERT conversation batches"""
    # Check if batch contains power features
    has_power = len(batch[0]) == 4
    
    if has_power:
        all_input_ids, all_attention_masks, all_powers, all_labels = [], [], [], []
        for input_ids, attention_mask, powers, labels in batch:
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_powers.append(powers)
            all_labels.append(labels)
    else:
        all_input_ids, all_attention_masks, all_labels = [], [], []
        for input_ids, attention_mask, labels in batch:
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)
    
    # Create conversation boundaries for the model
    conv_boundaries = [0]
    cumulative_len = 0
    for input_id_batch in all_input_ids:
        cumulative_len += len(input_id_batch)
        conv_boundaries.append(cumulative_len)
    
    # Concatenate all inputs across batches
    all_input_ids = torch.cat(all_input_ids)
    all_attention_masks = torch.cat(all_attention_masks)
    
    # For conversation-level tasks, take only the last label from each conversation
    conv_labels = [label_seq[-1] for label_seq in all_labels]  # Take last message's label
    all_labels = torch.stack(conv_labels)
    
    if has_power:
        all_powers = torch.cat(all_powers)
        return all_input_ids, all_attention_masks, conv_boundaries, all_powers, all_labels
    else:
        return all_input_ids, all_attention_masks, conv_boundaries, all_labels

class BERTContextModel(nn.Module):
    """BERT + Context model for deception detection"""
    def __init__(self, conv_hidden_dim=200, output_dim=2, use_power=False, dropout=0.3):
        super(BERTContextModel, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # BERT output dimension is 768
        bert_output_dim = 768
        
        # Conversation encoder (unidirectional LSTM)
        self.conv_lstm = nn.LSTM(bert_output_dim, 
                                conv_hidden_dim,
                                bidirectional=False,
                                batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Power feature flag
        self.use_power = use_power
        
        # Output layer
        if use_power:
            self.fc = nn.Linear(conv_hidden_dim + 2, output_dim)  # +2 for power features
        else:
            self.fc = nn.Linear(conv_hidden_dim, output_dim)
    
    def forward(self, input_ids, attention_mask, conv_boundaries, power=None):
        # Process messages with BERT
        # input_ids shape: [total_messages, seq_len]
        # attention_mask shape: [total_messages, seq_len]
        
        # Get BERT embeddings - use CLS token ([0]) for message representation
        with torch.no_grad():  # Freeze BERT weights
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            message_repr = bert_outputs.pooler_output  # [total_messages, 768]
        
        message_repr = self.dropout(message_repr)
        
        # Group messages by conversation
        conversation_inputs = []
        for i in range(len(conv_boundaries) - 1):
            start, end = conv_boundaries[i], conv_boundaries[i+1]
            conversation_inputs.append(message_repr[start:end])
        
        # Pad conversations to same length
        padded_convs = pad_sequence(conversation_inputs, batch_first=True)
        
        # Conversation-level encoding
        _, (conv_hidden, _) = self.conv_lstm(padded_convs)
        
        # Get final hidden state
        conv_hidden = conv_hidden.squeeze(0)  # [batch_size, conv_hidden_dim]
        conv_hidden = self.dropout(conv_hidden)
        
        # Add power features if requested
        if self.use_power and power is not None:
            # Process power features similar to the ContextLSTM implementation
            batch_powers = []
            for i in range(len(conv_boundaries) - 1):
                start, end = conv_boundaries[i], conv_boundaries[i+1]
                # Use the power of the last message in each conversation
                batch_powers.append(power[end - 1])
            batch_powers = torch.stack(batch_powers)
            
            # Combine with conversation representation
            combined = torch.cat((conv_hidden, batch_powers), dim=1)
            output = self.fc(combined)
        else:
            output = self.fc(conv_hidden)
        
        return output

def train_epoch(model, data_loader, optimizer, criterion, device, use_power=False):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    # Add progress bar for training
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        if use_power:
            input_ids, attention_mask, conv_boundaries, powers, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            powers = powers.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(input_ids, attention_mask, conv_boundaries, powers)
            
        else:
            input_ids, attention_mask, conv_boundaries, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(input_ids, attention_mask, conv_boundaries)
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Calculate accuracy
        preds = torch.argmax(predictions, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        # Update progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")
    
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

def evaluate(model, data_loader, criterion, device, use_power=False):
    """Evaluate model on a dataset"""
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    all_preds = []
    all_labels = []
    
    # Add progress bar for evaluation
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            if use_power:
                input_ids, attention_mask, conv_boundaries, powers, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                powers = powers.to(device)
                labels = labels.to(device)
                
                # Forward pass
                predictions = model(input_ids, attention_mask, conv_boundaries, powers)
            else:
                input_ids, attention_mask, conv_boundaries, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # Forward pass
                predictions = model(input_ids, attention_mask, conv_boundaries)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Calculate accuracy
            preds = torch.argmax(predictions, dim=1)
            acc = torch.sum(preds == labels).float() / len(labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")
    
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader), all_preds, all_labels

def run_bertcontext_model(task="sender", use_power=False, save_model=True):
    """Run BERT + Context model for deception detection"""
    # Load datasets
    print(f"Loading data for {task.upper()} task with power={use_power}")
    with jsonlines.open(train_path, 'r') as reader:
        train_data = list(reader)
    with jsonlines.open(val_path, 'r') as reader:
        val_data = list(reader)
    with jsonlines.open(test_path, 'r') as reader:
        test_data = list(reader)
    
    # Initialize BERT tokenizer
    print("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    max_length = 128  # BERT's standard max sequence length
    max_conv_length = 10  # Maximum conversation length
    
    print("Creating datasets...")
    train_dataset = ConversationBERTDataset(train_data, tokenizer, max_length, max_conv_length, use_power, task)
    val_dataset = ConversationBERTDataset(val_data, tokenizer, max_length, max_conv_length, use_power, task)
    test_dataset = ConversationBERTDataset(test_data, tokenizer, max_length, max_conv_length, use_power, task)
    
    # Create data loaders
    batch_size = 2  # Small batch size due to BERT's memory requirements
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_bert_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_bert_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_bert_batch)
    
    # Initialize model
    conv_hidden_dim = 200  # From the config file
    output_dim = 2  # Binary classification
    
    # Set dropout based on task (from config files)
    dropout = 0.4 if task.lower() == "sender" else 0.1
    
    print(f"Initializing BERT + Context model with dropout={dropout}...")
    model = BERTContextModel(
        conv_hidden_dim=conv_hidden_dim,
        output_dim=output_dim,
        use_power=use_power,
        dropout=dropout
    ).to(device)
    
    # Calculate class weights to handle class imbalance
    # Values from the config files
    if task.lower() == "sender":
        positive_weight = 15.0 if use_power else 15.0
    else:  # receiver task
        positive_weight = 10.0 if use_power else 20.0
    
    print(f"Using positive class weight: {positive_weight}")
    
    # Use weighted cross-entropy loss
    weight = torch.tensor([1.0, positive_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Initialize optimizer
    lr = 0.0003  # From the config file
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    n_epochs = 15  # From the config file
    patience = 10  # From the config file
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None
    best_val_metrics = None
    
    print(f"Starting training for {n_epochs} epochs...")
    # Add progress bar for epochs
    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, use_power)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device, use_power)
        
        # Calculate F1 scores
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
        val_binary_f1 = f1_score(val_labels, val_preds, average='binary', pos_label=1)
        
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}, Val Binary F1: {val_binary_f1:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model = model.state_dict().copy()
            best_val_metrics = {
                'loss': val_loss,
                'acc': val_acc,
                'macro_f1': val_macro_f1,
                'binary_f1': val_binary_f1,
                'epoch': epoch + 1
            }
            print(f"New best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model for testing
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device, use_power)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'macro_f1': f1_score(test_labels, test_preds, average='macro'),
        'binary_f1': f1_score(test_labels, test_preds, average='binary', pos_label=1),
        'precision': precision_score(test_labels, test_preds, pos_label=1, zero_division=0),
        'recall': recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    }
    
    # Save the best model if requested
    if save_model and best_model is not None:
        power_suffix = "_with_power" if use_power else "_without_power"
        model_filename = f"bertcontext_{task}{power_suffix}.pt"
        
        model_path = os.path.join(models_dir, model_filename)
        
        # Save model state and configuration
        torch.save({
            'model_state_dict': best_model,
            'conv_hidden_dim': conv_hidden_dim,
            'dropout': dropout,
            'val_metrics': best_val_metrics,
            'config': {
                'task': task,
                'use_power': use_power,
                'max_length': max_length,
                'max_conv_length': max_conv_length
            }
        }, model_path)
        
        print(f"\nBest model saved to {model_path}")
        print(f"Best validation metrics: Loss={best_val_metrics['loss']:.4f}, Macro F1={best_val_metrics['macro_f1']:.4f} (Epoch {best_val_metrics['epoch']})")
    
    # Print results
    print(f"\n=== BERT + Context Results for {task.upper()} Task ===")
    print(f"Power features: {'Yes' if use_power else 'No'}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Binary/Lie F1: {metrics['binary_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, digits=4, target_names=['Truthful', 'Deceptive']))
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BERT + Context model for deception detection')
    parser.add_argument('--task', choices=['sender', 'receiver'], default='sender',
                        help='Task to perform: "sender" for actual lie detection or "receiver" for suspected lie detection')
    parser.add_argument('--power', action='store_true', help='Use power features')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    args = parser.parse_args()
    
    # Run the model
    metrics = run_bertcontext_model(task=args.task, use_power=args.power, save_model=not args.no_save)
    
    # Compare with paper results
    paper_results = {
        'sender': {
            'with_power': {'macro_f1': 0.691, 'binary_f1': 0.527},
            'without_power': {'macro_f1': 0.672, 'binary_f1': 0.471},
        },
        'receiver': {
            'with_power': {'macro_f1': 0.749, 'binary_f1': 0.664},
            'without_power': {'macro_f1': 0.732, 'binary_f1': 0.630},
        }
    }
    
    power_key = 'with_power' if args.power else 'without_power'
    paper_macro_f1 = paper_results[args.task][power_key]['macro_f1']
    paper_binary_f1 = paper_results[args.task][power_key]['binary_f1']
    
    print("\n=== Comparison with Paper Results ===")
    print(f"Our Macro F1: {metrics['macro_f1']:.4f}   Paper: {paper_macro_f1:.4f}")
    print(f"Our Binary F1: {metrics['binary_f1']:.4f}   Paper: {paper_binary_f1:.4f}")
