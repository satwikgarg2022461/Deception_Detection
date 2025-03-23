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
import re
import argparse
import tqdm
from collections import Counter, defaultdict
from tqdm import tqdm

# For handling word embeddings
import gensim.downloader
from gensim.models import KeyedVectors
import pickle

# Set paths
project_dir = "d:\\NLP\\Deception_Detection"
data_dir = os.path.join(project_dir, "dataset")
models_dir = os.path.join(project_dir, "models")
test_path = os.path.join(data_dir, "test.jsonl")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ConversationDataset(Dataset):
    """Dataset for conversation-level processing"""
    def __init__(self, data, word_to_idx, max_length=100, max_conv_length=10, use_power=False, task="sender"):
        self.data = data
        self.word_to_idx = word_to_idx
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
            conv_messages = []
            conv_lengths = []
            conv_powers = []
            conv_labels = []
            
            for i, message in enumerate(conversation['messages']):
                # Process message text
                text = message
                tokens = tokenize_text(text)
                
                # Skip empty messages or ensure minimum length of 1
                if len(tokens) == 0:
                    tokens = ["<UNK>"]  # Use unknown token for empty messages
                    
                indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
                
                # Truncate if too long
                if len(indices) > self.max_length:
                    indices = indices[:self.max_length]
                    
                conv_lengths.append(len(indices))
                # Add padding to max_length
                indices = indices + [self.word_to_idx['<PAD>']] * (self.max_length - len(indices))
                conv_messages.append(indices)
                
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
            if len(conv_messages) == 0:
                continue
                
            # Truncate if too many messages
            if len(conv_messages) > self.max_conv_length:
                conv_messages = conv_messages[-self.max_conv_length:]
                conv_lengths = conv_lengths[-self.max_conv_length:]
                conv_labels = conv_labels[-self.max_conv_length:]
                if self.use_power:
                    conv_powers = conv_powers[-self.max_conv_length:]
            
            # Add to processed data
            if self.use_power:
                self.processed_data.append({
                    'messages': conv_messages,
                    'lengths': conv_lengths,
                    'powers': conv_powers,
                    'labels': conv_labels
                })
            else:
                self.processed_data.append({
                    'messages': conv_messages,
                    'lengths': conv_lengths,
                    'labels': conv_labels
                })
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        data = self.processed_data[idx]
        
        # Convert to tensors
        messages = torch.tensor(data['messages'], dtype=torch.long)
        lengths = torch.tensor(data['lengths'], dtype=torch.long)
        labels = torch.tensor(data['labels'], dtype=torch.long)
        
        if self.use_power:
            powers = torch.tensor(data['powers'], dtype=torch.float)
            return messages, lengths, powers, labels
        else:
            return messages, lengths, labels

def collate_batch(batch):
    """Custom collate function to handle variable-length conversations"""
    # Check if batch contains power features
    has_power = len(batch[0]) == 4
    
    if has_power:
        messages, lengths, powers, labels = zip(*batch)
        powers = torch.cat(powers)
    else:
        messages, lengths, labels = zip(*batch)
    
    # Flatten messages and lengths
    all_messages = torch.cat(messages)
    all_lengths = torch.cat(lengths)
    
    # For conversation-level tasks, we DON'T want to flatten labels
    # Instead keep them as batch of label lists
    # If we need the last label for each conversation:
    conv_labels = [conv_label[-1] for conv_label in labels]  # Take last message's label
    all_labels = torch.stack(conv_labels)
    
    # Store conversation boundaries for the conversation encoder
    conv_boundaries = [0]
    total_msgs = 0
    for msgs in messages:
        total_msgs += len(msgs)
        conv_boundaries.append(total_msgs)
    
    if has_power:
        return all_messages, all_lengths, conv_boundaries, powers, all_labels
    else:
        return all_messages, all_lengths, conv_boundaries, all_labels
    
class ContextLSTMModel(nn.Module):
    """Hierarchical LSTM model for deception detection"""
    def __init__(self, vocab_size, embedding_dim, message_hidden_dim, conv_hidden_dim, 
                 output_dim, embedding_weights=None, use_power=False, dropout=0.3):
        super(ContextLSTMModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight = nn.Parameter(embedding_weights)
        
        # Message encoder (bidirectional LSTM)
        self.message_lstm = nn.LSTM(embedding_dim, 
                                   message_hidden_dim, 
                                   bidirectional=True, 
                                   batch_first=True)
        
        # Conversation encoder (unidirectional LSTM)
        # Input size is 2*message_hidden_dim due to bidirectional message LSTM
        self.conv_lstm = nn.LSTM(2 * message_hidden_dim, 
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
    
    def forward(self, messages, lengths, conv_boundaries, power=None):
        # messages shape: [total_messages, seq_len]
        
        # Get embeddings
        embedded = self.embedding(messages)  # [total_messages, seq_len, embedding_dim]
        
        # Message-level encoding
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.message_lstm(packed_embedded)
        
        # Concatenate forward and backward hidden states
        # hidden shape: [2, total_messages, message_hidden_dim]
        message_repr = torch.cat((hidden[0], hidden[1]), dim=1)  # [total_messages, 2*message_hidden_dim]
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
        # For this, we need to associate power with the right message
        # Since we're predicting at conversation level, we'll need to select the power for the target message
        if self.use_power and power is not None:
            # Here we could select power features for the last message in each conversation
            # or use an attention mechanism to focus on specific messages
            # For simplicity, we'll use the power of the last message in each conversation
            batch_powers = []
            for i in range(len(conv_boundaries) - 1):
                start, end = conv_boundaries[i], conv_boundaries[i+1]
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
    
    # Add progress bar for training batches
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        if use_power:
            messages, lengths, conv_boundaries, powers, labels = batch
            messages = messages.to(device)
            lengths = lengths.to(device)
            powers = powers.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(messages, lengths, conv_boundaries, powers)
            
        else:
            messages, lengths, conv_boundaries, labels = batch
            messages = messages.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(messages, lengths, conv_boundaries)
        
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
                messages, lengths, conv_boundaries, powers, labels = batch
                messages = messages.to(device)
                lengths = lengths.to(device)
                powers = powers.to(device)
                labels = labels.to(device)
                
                # Forward pass
                predictions = model(messages, lengths, conv_boundaries, powers)
            else:
                messages, lengths, conv_boundaries, labels = batch
                messages = messages.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                
                # Forward pass
                predictions = model(messages, lengths, conv_boundaries)
            
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

def run_contextlstm_model(task="sender", use_power=False, save_model=True):
    """Run ContextLSTM model for deception detection"""
    # Load datasets
    print(f"Loading data for {task.upper()} task with power={use_power}")
    with jsonlines.open(train_path, 'r') as reader:
        train_data = list(reader)
    with jsonlines.open(val_path, 'r') as reader:
        val_data = list(reader)
    with jsonlines.open(test_path, 'r') as reader:
        test_data = list(reader)
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    word_counts = Counter()
    for conversation in tqdm(train_data, desc="Processing conversations"):
        for message in conversation['messages']:
            tokens = tokenize_text(message)
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
    max_conv_length = 10  # Maximum conversation length
    train_dataset = ConversationDataset(train_data, word_to_idx, max_length, max_conv_length, use_power, task)
    val_dataset = ConversationDataset(val_data, word_to_idx, max_length, max_conv_length, use_power, task)
    test_dataset = ConversationDataset(test_data, word_to_idx, max_length, max_conv_length, use_power, task)
    
    # Create data loaders
    batch_size = 4  # From the config file (reduced due to hierarchical nature)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)
    
    # Initialize model
    message_hidden_dim = 100  # From the config file
    conv_hidden_dim = 200  # From the config file
    output_dim = 2  # Binary classification
    
    # Set dropout based on task (from config files)
    dropout = 0.3 if task.lower() == "sender" else 0.4
    
    model = ContextLSTMModel(
        vocab_size=len(word_to_idx), 
        embedding_dim=embedding_dim,
        message_hidden_dim=message_hidden_dim,
        conv_hidden_dim=conv_hidden_dim,
        output_dim=output_dim,
        embedding_weights=embedding_weights,
        use_power=use_power,
        dropout=dropout
    ).to(device)
    
    # Calculate class weights to handle class imbalance (from config files)
    if task.lower() == "sender":
        positive_weight = 10.0  # From actual_lie config
    else:  # receiver task
        positive_weight = 15.0  # From suspected_lie config
    
    # Use weighted cross-entropy loss
    weight = torch.tensor([1.0, positive_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Initialize optimizer
    lr = 0.003  # From the config file
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    n_epochs = 15  # From the config file
    patience = 10  # From the config file for contextlstm
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
        model_filename = f"contextlstm_{task}{power_suffix}.pt"
        vocab_filename = f"contextlstm_vocab_{task}{power_suffix}.pkl"
        
        model_path = os.path.join(models_dir, model_filename)
        vocab_path = os.path.join(models_dir, vocab_filename)
        
        # Save model state and vocabulary
        torch.save({
            'model_state_dict': best_model,
            'vocab_size': len(word_to_idx),
            'embedding_dim': embedding_dim,
            'message_hidden_dim': message_hidden_dim,
            'conv_hidden_dim': conv_hidden_dim,
            'val_metrics': best_val_metrics,
            'config': {
                'task': task,
                'use_power': use_power,
                'max_length': max_length,
                'max_conv_length': max_conv_length,
                'dropout': dropout
            }
        }, model_path)
        
        # Save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(word_to_idx, f)
        
        print(f"\nBest model saved to {model_path}")
        print(f"Vocabulary saved to {vocab_path}")
        print(f"Best validation metrics: Loss={best_val_metrics['loss']:.4f}, Macro F1={best_val_metrics['macro_f1']:.4f} (Epoch {best_val_metrics['epoch']})")
    
    # Print results
    print(f"\n=== ContextLSTM Results for {task.upper()} Task ===")
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
    parser = argparse.ArgumentParser(description='Run ContextLSTM model for deception detection')
    parser.add_argument('--task', choices=['sender', 'receiver'], default='sender',
                        help='Task to perform: "sender" for actual lie detection or "receiver" for suspected lie detection')
    parser.add_argument('--power', action='store_true', help='Use power features')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    args = parser.parse_args()
    
    # Run the model
    metrics = run_contextlstm_model(task=args.task, use_power=args.power, save_model=not args.no_save)
    
    # Compare with paper results
    paper_results = {
        'sender': {
            'with_power': {'macro_f1': 0.656, 'binary_f1': 0.460},
            'without_power': {'macro_f1': 0.625, 'binary_f1': 0.400},
        },
        'receiver': {
            'with_power': {'macro_f1': 0.711, 'binary_f1': 0.606},
            'without_power': {'macro_f1': 0.682, 'binary_f1': 0.568},
        }
    }
    
    power_key = 'with_power' if args.power else 'without_power'
    paper_macro_f1 = paper_results[args.task][power_key]['macro_f1']
    paper_binary_f1 = paper_results[args.task][power_key]['binary_f1']
    
    print("\n=== Comparison with Paper Results ===")
    print(f"Our Macro F1: {metrics['macro_f1']:.4f}   Paper: {paper_macro_f1:.4f}")
    print(f"Our Binary F1: {metrics['binary_f1']:.4f}   Paper: {paper_binary_f1:.4f}")
