import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import jsonlines
import os
import re
import argparse
import tqdm
from collections import Counter

# For handling word embeddings
import gensim.downloader
from gensim.models import KeyedVectors
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

# Set paths
project_dir = "d:\\NLP\\Deception_Detection"
data_dir = os.path.join(project_dir, "dataset")
test_path = os.path.join(data_dir, "test.jsonl")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert conversations into single messages (same as original)
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

# Tokenization function
def tokenize_text(text):
    """Simple tokenization function for text"""
    # Convert to lowercase
    text = text.lower()
    # Replace numbers with special token
    text = re.sub(r'\d+', '_NUM_', text)
    # Simple tokenization by splitting on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

class MessageDataset(Dataset):
    """Dataset for message classification"""
    def __init__(self, messages, word_to_idx, max_length=100, use_power=False, task="sender"):
        self.messages = messages
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.use_power = use_power
        self.task = task
        
        # Filter messages based on task
        self.filtered_messages = []
        for msg in messages:
            if task.lower() == "receiver" and msg['receiver_annotation'] not in [True, False]:
                continue  # Skip messages without receiver annotation for receiver task
            self.filtered_messages.append(msg)
    
    def __len__(self):
        return len(self.filtered_messages)
    
    def __getitem__(self, idx):
        message = self.filtered_messages[idx]
        text = message['message']
        
        # Tokenize and convert to indices
        tokens = tokenize_text(text)
        indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        
        # Pad or truncate sequence
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.word_to_idx['<PAD>']] * (self.max_length - len(indices))
        
        # Create tensor from indices
        input_tensor = torch.tensor(indices, dtype=torch.long)
        
        # Create label tensor based on task
        if self.task.lower() == "sender":
            label = 0 if message['sender_annotation'] else 1  # 0 = truthful, 1 = deceptive
        else:  # receiver task
            label = 0 if message['receiver_annotation'] else 1  # 0 = truthful, 1 = deceptive
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Add power feature if requested
        if self.use_power:
            # Binary power features (severe power imbalance indicators)
            power_tensor = torch.tensor([
                1 if message['score_delta'] > 4 else 0,  # Sender much stronger
                1 if message['score_delta'] < -4 else 0   # Receiver much stronger
            ], dtype=torch.float)
            
            return input_tensor, power_tensor, label_tensor
        
        return input_tensor, label_tensor

class LSTMModel(nn.Module):
    """LSTM model for deception detection"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 embedding_weights=None, use_power=False, dropout=0.5):
        super(LSTMModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight = nn.Parameter(embedding_weights)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           bidirectional=True, 
                           batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Power feature flag
        self.use_power = use_power
        
        # Output layer
        if use_power:
            self.fc = nn.Linear(hidden_dim * 2 + 2, output_dim)  # *2 for bidirectional, +2 for power features
        else:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
    
    def forward(self, text, power=None):
        # text shape: [batch_size, seq_len]
        
        # Get embeddings
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        # Add power features if requested
        if self.use_power and power is not None:
            hidden = torch.cat((hidden, power), dim=1)
        
        # Pass through linear layer
        return self.fc(hidden)

def load_glove_embeddings(word_to_idx, embedding_dim=200):
    """Load GloVe Twitter embeddings"""
    print(f"Loading GloVe Twitter embeddings (dimension={embedding_dim})...")
    
    # Path for cached embeddings
    cache_dir = os.path.join(project_dir, "embeddings_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"glove_twitter_{embedding_dim}d.pkl")
    
    # Try to load cached embeddings
    if os.path.exists(cache_file):
        print(f"Loading embeddings from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            glove_model = pickle.load(f)
    else:
        print(f"Downloading GloVe Twitter embeddings...")
        try:
            # Load glove twitter embeddings
            glove_model = gensim.downloader.load('glove-twitter-200')
            
            # Cache the embeddings for future use
            with open(cache_file, "wb") as f:
                pickle.dump(glove_model, f)
        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            return None
    
    # Create embedding matrix
    embedding_matrix = torch.zeros((len(word_to_idx), embedding_dim))
    
    # Initialize with random values for unknown words
    for word, idx in word_to_idx.items():
        if word in glove_model:
            embedding_matrix[idx] = torch.tensor(glove_model[word], dtype=torch.float)
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim) * 0.1
    
    return embedding_matrix

def train_epoch(model, data_loader, optimizer, criterion, device, use_power=False):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    # Add progress bar for batches
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        if use_power:
            text, power, labels = batch
            text = text.to(device)
            power = power.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(text, power)
            
        else:
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(text)
        
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
                text, power, labels = batch
                text = text.to(device)
                power = power.to(device)
                labels = labels.to(device)
                
                # Forward pass
                predictions = model(text, power)
            else:
                text, labels = batch
                text = text.to(device)
                labels = labels.to(device)
                
                # Forward pass
                predictions = model(text)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Calculate accuracy
            preds = torch.argmax(predictions, dim=1)
            acc = torch.sum(preds == labels).float() / len(labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader), all_preds, all_labels

def run_lstm_model(task="sender", use_power=False, save_model=True):
    """Run LSTM model for deception detection"""
    # Load datasets
    print(f"Loading data for {task.upper()} task with power={use_power}")
    with jsonlines.open(train_path, 'r') as reader:
        train_data = list(reader)
    with jsonlines.open(val_path, 'r') as reader:
        val_data = list(reader)
    with jsonlines.open(test_path, 'r') as reader:
        test_data = list(reader)
    
    # Process data
    train_messages = aggregate(train_data)
    val_messages = aggregate(val_data)
    test_messages = aggregate(test_data)
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    word_counts = Counter()
    for message in tqdm(train_messages, desc="Counting words"):
        tokens = tokenize_text(message['message'])
        word_counts.update(tokens)
    
    # Keep only words appearing at least 3 times
    min_freq = 3
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create word to index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for i, word in enumerate(vocab, 2):  # Start from 2 as 0 and 1 are reserved
        word_to_idx[word] = i
    
    # Load GloVe embeddings
    embedding_dim = 200  # From the config file
    embedding_weights = load_glove_embeddings(word_to_idx, embedding_dim)
    
    # Create datasets
    max_length = 100  # Maximum sequence length
    train_dataset = MessageDataset(train_messages, word_to_idx, max_length, use_power, task)
    val_dataset = MessageDataset(val_messages, word_to_idx, max_length, use_power, task)
    test_dataset = MessageDataset(test_messages, word_to_idx, max_length, use_power, task)
    
    # Create data loaders
    batch_size = 32  # From the config file
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    hidden_dim = 100  # From the config file
    output_dim = 2  # Binary classification
    dropout = 0.5  # From the config file
    
    model = LSTMModel(
        vocab_size=len(word_to_idx), 
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim, 
        output_dim=output_dim,
        embedding_weights=embedding_weights,
        use_power=use_power,
        dropout=dropout
    ).to(device)
    
    # Calculate class weights to handle class imbalance
    if task.lower() == "sender":
        positive_weight = 30.0  # posclass_weight from config file
    else:  # receiver task
        positive_weight = 30.0  # Same weight for simplicity
    
    # Use weighted cross-entropy loss
    weight = torch.tensor([1.0, positive_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Initialize optimizer
    lr = 0.003  # From the config file
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    n_epochs = 15  # From the config file
    patience = 5  # From the config file
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
    
    # Print results
    print(f"\n=== LSTM Results for {task.upper()} Task ===")
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
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the best model if requested
    if save_model and best_model is not None:
        power_suffix = "_with_power" if use_power else "_without_power"
        model_filename = f"lstm_{task}{power_suffix}.pt"
        vocab_filename = f"lstm_vocab_{task}{power_suffix}.pkl"
        
        model_path = os.path.join(models_dir, model_filename)
        vocab_path = os.path.join(models_dir, vocab_filename)
        
        # Save model state and vocabulary
        torch.save({
            'model_state_dict': best_model,
            'vocab_size': len(word_to_idx),
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'val_metrics': best_val_metrics,
            'config': {
                'task': task,
                'use_power': use_power,
                'max_length': max_length,
                'dropout': dropout
            }
        }, model_path)
        
        # Save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(word_to_idx, f)
        
        print(f"\nBest model saved to {model_path}")
        print(f"Vocabulary saved to {vocab_path}")
        print(f"Best validation metrics: Loss={best_val_metrics['loss']:.4f}, Macro F1={best_val_metrics['macro_f1']:.4f} (Epoch {best_val_metrics['epoch']})")
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSTM model for deception detection')
    parser.add_argument('--task', choices=['sender', 'receiver'], default='sender',
                        help='Task to perform: "sender" for actual lie detection or "receiver" for suspected lie detection')
    parser.add_argument('--power', action='store_true', help='Use power features')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    args = parser.parse_args()
    
    # Run the model
    metrics = run_lstm_model(task=args.task, use_power=args.power, save_model=not args.no_save)
    
    # Compare with paper results
    paper_results = {
        'sender': {
            'with_power': {'macro_f1': 0.625, 'binary_f1': 0.400},
            'without_power': {'macro_f1': 0.622, 'binary_f1': 0.394},
        },
        'receiver': {
            'with_power': {'macro_f1': 0.682, 'binary_f1': 0.568},
            'without_power': {'macro_f1': 0.678, 'binary_f1': 0.562},
        }
    }
    
    power_key = 'with_power' if args.power else 'without_power'
    paper_macro_f1 = paper_results[args.task][power_key]['macro_f1']
    paper_binary_f1 = paper_results[args.task][power_key]['binary_f1']
    
    print("\n=== Comparison with Paper Results ===")
    print(f"Our Macro F1: {metrics['macro_f1']:.4f}   Paper: {paper_macro_f1:.4f}")
    print(f"Our Binary F1: {metrics['binary_f1']:.4f}   Paper: {paper_binary_f1:.4f}")
