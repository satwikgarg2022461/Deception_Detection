


# Still in development






import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


# Tokenization function - similar to WordTokenizer in AllenNLP
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
    """Load GloVe Twitter embeddings - matches config file's pretrained_file"""
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
            # Load glove twitter embeddings - specifically glove.twitter.27B.200d
            # This matches the config file: "pretrained_file": "(http://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt"
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


# Fix the weighted loss function to handle dimensions correctly
def weighted_sequence_cross_entropy_with_logits(logits, targets, mask, weights=None):
    """Implementation of weighted_sequence_cross_entropy_with_logits from hlstm.py"""
    # Flatten all dimensions for loss calculation
    batch_size, seq_len, num_classes = logits.size()

    # Reshape logits to (batch_size*seq_len, num_classes)
    logits_flat = logits.view(-1, num_classes)

    # Ensure mask is 1D and same length as flattened logits
    mask_flat = mask.view(-1).float()

    # Ensure targets is 1D and has the same length as mask
    if targets.dim() == 1:
        # If targets is already 1D, just make sure it aligns with valid positions
        valid_indices = torch.nonzero(mask_flat).squeeze()
        if len(valid_indices.shape) == 0:
            valid_indices = valid_indices.unsqueeze(0)
        valid_logits = logits_flat[valid_indices]
        valid_targets = targets  # Already correct size
    else:
        # If targets is 2D or higher, flatten it
        targets_flat = targets.view(-1)
        valid_indices = torch.nonzero(mask_flat).squeeze()
        if len(valid_indices.shape) == 0:
            valid_indices = valid_indices.unsqueeze(0)
        valid_logits = logits_flat[valid_indices]
        valid_targets = targets_flat[valid_indices]

    # Calculate element-wise loss with specified weights
    loss = F.cross_entropy(valid_logits, valid_targets, weight=weights, reduction='mean')
    return loss


class ConversationDataset(Dataset):
    """Dataset for conversation-level processing - equivalent to diplomacy_reader from game_reader.py"""

    def __init__(self, data, word_to_idx, max_length=100, max_conv_length=10, use_power=False, task="sender"):
        self.data = data
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.max_conv_length = max_conv_length
        self.use_power = use_power
        self.task = task  # Corresponds to 'label_key' in diplomacy_reader

        # Process conversations
        self.processed_data = []
        self.process_conversations()

    def process_conversations(self):
        """Process conversations to match DiplomacyReader._read implementation"""
        for conversation in self.data:
            # Skip empty conversations
            if len(conversation['messages']) == 0:
                continue

            conv_messages = []
            conv_lengths = []
            conv_powers = []
            conv_labels = []
            conv_speakers = []

            # Get label key based on task
            label_key = 'sender_labels' if self.task.lower() == "sender" else 'receiver_labels'

            # Extract messages, speakers, and labels (matching DiplomacyReader.text_to_instance)
            for i, (message, speaker) in enumerate(zip(conversation['messages'], conversation['speakers'])):
                # Skip messages with invalid labels
                if label_key not in conversation or i >= len(conversation[label_key]):
                    continue

                label = conversation[label_key][i]
                if label not in [True, False]:
                    continue

                # Process message text
                tokens = tokenize_text(message)

                # Skip empty messages or ensure minimum length of 1
                if len(tokens) == 0:
                    tokens = ["<UNK>"]

                indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]

                # Truncate if too long
                if len(indices) > self.max_length:
                    indices = indices[:self.max_length]

                # Following DiplomacyReader, True = truthful (0), False = deceptive (1)
                binary_label = 0 if label else 1

                conv_lengths.append(len(indices))
                # Add padding to max_length
                indices = indices + [self.word_to_idx['<PAD>']] * (self.max_length - len(indices))
                conv_messages.append(indices)
                conv_labels.append(binary_label)
                conv_speakers.append(speaker)

                # Process power features if requested - matches use_game_scores in config
                if self.use_power:
                    game_score = conversation['game_score_delta'][i]
                    # Store raw game score as in DiplomacyReader
                    conv_powers.append([float(game_score)])

            # Skip if no valid messages in this conversation
            if len(conv_messages) == 0:
                continue

            # Truncate if too many messages
            if len(conv_messages) > self.max_conv_length:
                conv_messages = conv_messages[-self.max_conv_length:]
                conv_lengths = conv_lengths[-self.max_conv_length:]
                conv_labels = conv_labels[-self.max_conv_length:]
                conv_speakers = conv_speakers[-self.max_conv_length:]
                if self.use_power:
                    conv_powers = conv_powers[-self.max_conv_length:]

            # Add to processed data
            data_item = {
                'messages': conv_messages,
                'lengths': conv_lengths,
                'labels': conv_labels,
                'speakers': conv_speakers
            }

            if self.use_power:
                data_item['powers'] = conv_powers

            self.processed_data.append(data_item)

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

    # Convert batch of labels to tensor
    all_labels = torch.cat(labels)

    # Create conversation mask - equivalent to mask in hlstm.py
    conversation_mask = []
    for msg in messages:
        conversation_mask.append(torch.ones(len(msg)))
    conversation_mask = torch.cat(conversation_mask)

    # Store conversation boundaries for the conversation encoder
    conv_boundaries = [0]
    total_msgs = 0
    for msgs in messages:
        total_msgs += len(msgs)
        conv_boundaries.append(total_msgs)

    if has_power:
        return all_messages, all_lengths, conversation_mask, conv_boundaries, powers, all_labels
    else:
        return all_messages, all_lengths, conversation_mask, conv_boundaries, all_labels


class PooledRNN(nn.Module):
    """Implementation of pooled_rnn from pooled_rnn.py"""

    def __init__(self, input_size, hidden_size, bidirectional=True, poolers="max"):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.poolers = poolers.split(',')
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

    def forward(self, embedded, mask):
        """Implementation following PooledRNN.forward in the original pooled_rnn.py"""
        # Get lengths from mask for pack_padded_sequence
        lengths = mask.sum(dim=1).clamp(min=1).cpu()

        # Run through the RNN
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Get proper boolean mask
        mask = mask.bool()
        batch_size = embedded.size(0)
        pooled = []

        # Apply pooling based on poolers using approaches from pooled_rnn.py
        if 'max' in self.poolers:
            # Create a proper mask for max pooling (with -inf for padding)
            # Make sure the mask is properly reshaped to match output
            max_mask = mask.unsqueeze(-1).expand(-1, -1, output.size(-1))
            # For any positions beyond the actual output length, set mask to False
            if max_mask.size(1) > output.size(1):
                max_mask = max_mask[:, :output.size(1), :]

            # Apply the mask and get max values
            masked_output = output.masked_fill(~max_mask, float('-inf'))
            max_pooled, _ = torch.max(masked_output, dim=1)
            pooled.append(max_pooled)

        if 'mean' in self.poolers:
            # Create proper mask for mean pooling
            mean_mask = mask.unsqueeze(-1).float()

            # Ensure the mask matches output dimensions
            if mean_mask.size(1) > output.size(1):
                mean_mask = mean_mask[:, :output.size(1), :]
            else:
                mean_mask = mean_mask.expand(-1, -1, output.size(-1))

            # Sum values and divide by actual length
            sum_pooled = torch.sum(output * mean_mask, dim=1)
            mean_pooled = sum_pooled / lengths.float().unsqueeze(-1).clamp(min=1)
            pooled.append(mean_pooled)

        if 'last' in self.poolers:
            # Extract last valid state as in the original pooled_rnn.py
            if not self.bidirectional:
                # For unidirectional, just use the last hidden state
                pooled.append(hidden[-1])
            else:
                # For bidirectional, combine first and last directions
                forward = hidden[0]  # Forward direction
                backward = hidden[1]  # Backward direction
                pooled.append(torch.cat([forward, backward], dim=1))

        # Concatenate all pooling results
        return torch.cat(pooled, dim=1) if len(pooled) > 1 else pooled[0]


class ContextLSTMModel(nn.Module):
    """Hierarchical LSTM model for deception detection - implements hierarchical_lstm from hlstm.py"""

    def __init__(self, vocab_size, embedding_dim, message_hidden_dim, conv_hidden_dim,
                 output_dim, embedding_weights=None, use_power=False, dropout=0.3):
        super().__init__()

        # Embedding layer - matches config's "embedder" with trainable=false
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight = nn.Parameter(embedding_weights)
            # Don't train the embeddings to match trainable=false
            self.embedding.weight.requires_grad = False

        # Message encoder using PooledRNN with max pooler - matches message_encoder in config
        self.message_encoder = PooledRNN(
            input_size=embedding_dim,
            hidden_size=message_hidden_dim,
            bidirectional=True,
            poolers="max"  # Match the config's "poolers": "max"
        )

        # Conversation encoder (unidirectional LSTM) - matches conversation_encoder in config
        message_output_dim = message_hidden_dim * 2  # Bidirectional doubles output size
        self.conv_lstm = nn.LSTM(
            input_size=message_output_dim,
            hidden_size=conv_hidden_dim,
            bidirectional=False,
            batch_first=True
        )

        # Dropout layer - matches "dropout" in config
        self.dropout = nn.Dropout(dropout)

        # Power feature flag - matches "use_game_scores" in config
        self.use_power = use_power

        # Output layer
        final_dim = conv_hidden_dim + (1 if use_power else 0)  # +1 for game score
        self.classifier = nn.Linear(final_dim, output_dim)

        # F1 metrics for evaluation
        self.output_dim = output_dim

    def forward(self, messages, lengths, mask, conv_boundaries, power=None):
        """Forward pass matching HierarchicalLSTM.forward in hlstm.py"""
        # Get embeddings
        embedded = self.embedding(messages)  # [total_messages, seq_len, embedding_dim]
        embedded = self.dropout(embedded)

        # Create a message mask that exactly matches the embedded dimensions
        # This is important to ensure proper masking in PooledRNN
        message_mask = torch.zeros((messages.size(0), embedded.size(1)),
                                   dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            message_mask[i, :length] = True

        # Message-level encoding
        message_repr = self.message_encoder(embedded, message_mask)
        message_repr = self.dropout(message_repr)

        # Group messages by conversation
        conversation_inputs = []
        conversation_masks = []
        batch_size = len(conv_boundaries) - 1
        for i in range(batch_size):
            start, end = conv_boundaries[i], conv_boundaries[i + 1]
            conversation_inputs.append(message_repr[start:end])
            # Create mask for valid messages in conversation
            msg_mask = torch.ones(end - start, dtype=torch.bool, device=device)
            conversation_masks.append(msg_mask)

        # Pad conversations to same length
        padded_convs = pad_sequence(conversation_inputs, batch_first=True)
        conversation_mask = pad_sequence(conversation_masks, batch_first=True, padding_value=False)

        # Conversation-level encoding
        packed_convs = pack_padded_sequence(
            padded_convs,
            lengths=[len(c) for c in conversation_inputs],
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, (conv_hidden, _) = self.conv_lstm(packed_convs)
        conv_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout to output
        conv_output = self.dropout(conv_output)

        # Add power features if requested - matches use_game_scores in config
        if self.use_power and power is not None:
            # Create a properly sized tensor for game scores
            # Must be same batch size and sequence length as conv_output
            batch_size = conv_output.size(0)
            seq_len = conv_output.size(1)
            game_scores_tensor = torch.zeros(batch_size, seq_len, 1, device=device)

            # Fill in the actual power values
            for i in range(len(conv_boundaries) - 1):
                start, end = conv_boundaries[i], conv_boundaries[i + 1]
                # Get power values for this conversation
                conv_powers = power[start:end]

                # Only fill up to the minimum of available scores or sequence length
                length = min(len(conv_powers), seq_len)
                if length > 0:
                    # Reshape to [length, 1] and place in the tensor
                    game_scores_tensor[i, :length, 0] = conv_powers[:length, 0]

            # Concatenate with conv_output along feature dimension
            conv_output = torch.cat([conv_output, game_scores_tensor], dim=2)

        # Apply classifier to get logits
        logits = self.classifier(conv_output)

        return logits, conversation_mask


def train_epoch(model, data_loader, optimizer, criterion, device, use_power=False):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0

    all_preds = []
    all_labels = []

    # Add progress bar for training batches
    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        if use_power:
            messages, lengths, mask, conv_boundaries, powers, labels = batch
            messages = messages.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            powers = powers.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits, conversation_mask = model(messages, lengths, mask, conv_boundaries, powers)

        else:
            messages, lengths, mask, conv_boundaries, labels = batch
            messages = messages.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits, conversation_mask = model(messages, lengths, mask, conv_boundaries)

        # Calculate loss using weighted_sequence_cross_entropy_with_logits from hlstm.py
        weight = torch.tensor([1.0, float(criterion.pos_weight)], device=device)

        # Make sure conversation_mask and labels are on the same device
        conversation_mask = conversation_mask.to(device)

        # Only keep the actual length of the labels array
        # This ensures we don't include padding labels
        total_length = 0
        for i in range(len(conv_boundaries) - 1):
            start, end = conv_boundaries[i], conv_boundaries[i + 1]
            total_length += end - start

        # Ensure labels match the actual conversation mask
        labels = labels[:total_length]

        # Calculate loss on correctly aligned tensors
        loss = weighted_sequence_cross_entropy_with_logits(logits, labels, conversation_mask, weights=weight)

        # Backward pass
        loss.backward()

        # Apply gradient clipping at 1.0 to match config's "grad_clipping": 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        optimizer.step()

        # Calculate accuracy and F1 - Fix the dimension mismatch here
        preds = torch.argmax(logits, dim=-1)

        # Extract valid predictions and labels - fix dimension issue
        valid_preds = []
        valid_labels = []

        # Properly align predictions and labels using conversation boundaries
        for i in range(len(conv_boundaries) - 1):
            start, end = conv_boundaries[i], conv_boundaries[i + 1]
            # Get mask for this conversation
            conv_mask = conversation_mask[i, :end - start]
            # Get predictions for this conversation where mask is True
            conv_preds = preds[i, :end - start][conv_mask]
            # Get corresponding labels
            conv_labels = labels[start:end]

            valid_preds.append(conv_preds)
            valid_labels.append(conv_labels)

        # Flatten predictions and labels
        valid_preds = torch.cat(valid_preds)
        valid_labels = torch.cat(valid_labels)

        # Update tracking
        all_preds.extend(valid_preds.cpu().numpy())
        all_labels.extend(valid_labels.cpu().numpy())

        epoch_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Calculate epoch metrics
    if len(all_preds) > 0:  # Avoid empty predictions
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_macro_f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        epoch_acc = 0.0
        epoch_macro_f1 = 0.0

    return epoch_loss / len(data_loader), epoch_acc, epoch_macro_f1


def evaluate(model, data_loader, criterion, device, use_power=False):
    """Evaluate model on a dataset"""
    model.eval()
    epoch_loss = 0

    all_preds = []
    all_labels = []

    # Add progress bar for evaluation
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            if use_power:
                messages, lengths, mask, conv_boundaries, powers, labels = batch
                messages = messages.to(device)
                lengths = lengths.to(device)
                mask = mask.to(device)
                powers = powers.to(device)
                labels = labels.to(device)

                # Forward pass
                logits, conversation_mask = model(messages, lengths, mask, conv_boundaries, powers)

            else:
                messages, lengths, mask, conv_boundaries, labels = batch
                messages = messages.to(device)
                lengths = lengths.to(device)
                mask = mask.to(device)
                labels = labels.to(device)

                # Forward pass
                logits, conversation_mask = model(messages, lengths, mask, conv_boundaries)

            # Calculate loss
            weight = torch.tensor([1.0, float(criterion.pos_weight)], device=device)
            loss = weighted_sequence_cross_entropy_with_logits(logits, labels, conversation_mask, weights=weight)

            # Calculate predictions - Fix the dimension mismatch here
            preds = torch.argmax(logits, dim=-1)

            # Extract valid predictions and labels - fix dimension issue
            valid_preds = []
            valid_labels = []

            # Properly align predictions and labels using conversation boundaries
            for i in range(len(conv_boundaries) - 1):
                start, end = conv_boundaries[i], conv_boundaries[i + 1]
                # Get mask for this conversation
                conv_mask = conversation_mask[i, :end - start]
                # Get predictions for this conversation where mask is True
                conv_preds = preds[i, :end - start][conv_mask]
                # Get corresponding labels
                conv_labels = labels[start:end]

                valid_preds.append(conv_preds)
                valid_labels.append(conv_labels)

            # Flatten predictions and labels
            if valid_preds:  # Check if we have any valid predictions
                valid_preds = torch.cat(valid_preds)
                valid_labels = torch.cat(valid_labels)

                # Update tracking
                all_preds.extend(valid_preds.cpu().numpy())
                all_labels.extend(valid_labels.cpu().numpy())

            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Calculate metrics
    metrics = {
        'loss': epoch_loss / len(data_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'binary_f1': f1_score(all_labels, all_preds, average='binary', pos_label=1),
        'precision': precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    }

    return metrics, all_preds, all_labels


class WeightedCrossEntropyLoss:
    """Custom loss class to match hlstm.py's weighted_sequence_cross_entropy_with_logits"""

    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight


# Fix the print statement that has uppercase UPPER() method instead of upper()
def run_contextlstm_model(task="sender", use_power=False, save_model=True):
    """Run ContextLSTM model for deception detection"""
    # Set random seeds for reproducibility - matching config's seeds
    torch.manual_seed(1994)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1994)

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

    # Analyze class distribution
    train_labels = []
    for item in train_dataset.processed_data:
        train_labels.extend(item['labels'])

    # Count labels
    label_counts = Counter(train_labels)
    print(f"Training data label distribution: {label_counts}")

    # Adjust positive class weight based on class imbalance - from configs
    if task.lower() == "sender":
        if use_power:
            positive_weight = 10.0  # From actual_lie/contextlstm+power.jsonnet
        else:
            positive_weight = 10.0  # From actual_lie/contextlstm.jsonnet
    else:  # receiver task
        if use_power:
            positive_weight = 10.0  # From suspected_lie/contextlstm+power.jsonnet
        else:
            positive_weight = 15.0  # From suspected_lie/contextlstm.jsonnet

    # Create data loaders
    batch_size = 4  # From the config file
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

    # Use weighted cross-entropy loss - matches pos_weight in config
    criterion = WeightedCrossEntropyLoss(pos_weight=positive_weight)

    # Initialize optimizer - matches config's "optimizer"
    lr = 0.003  # From the config file
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    n_epochs = 15  # From the config file
    patience = 10  # From the config file for contextlstm
    epochs_without_improvement = 0
    best_model = None
    best_val_metrics = None

    print(f"Starting training for {n_epochs} epochs with positive weight={positive_weight}...")
    # Add progress bar for epochs
    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        # Train
        train_loss, train_acc, train_macro_f1 = train_epoch(model, train_loader, optimizer, criterion, device,
                                                            use_power)

        # Validate
        val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, device, use_power)

        print(
            f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val Macro F1: {val_metrics['macro_f1']:.4f}, Val Binary F1: {val_metrics['binary_f1']:.4f}")

        # Check for improvement - use macro_f1 to match config's "validation_metric": "+macro_fscore"
        if best_val_metrics is None or val_metrics['macro_f1'] > best_val_metrics['macro_f1']:
            epochs_without_improvement = 0
            best_model = model.state_dict().copy()
            best_val_metrics = val_metrics.copy()
            best_val_metrics['epoch'] = epoch + 1
            print(f"New best model saved! (macro F1: {val_metrics['macro_f1']:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        # Early stopping with patience - matches config's "patience": 10
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # Load best model for testing
    if best_model is not None:
        model.load_state_dict(best_model)

    # Evaluate on test set - matches config's "evaluate_on_test": true
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device, use_power)

    # Print prediction distribution
    pred_counts = Counter(test_preds)
    label_counts = Counter(test_labels)
    print("\nPrediction distribution:")
    print(f"Predicted labels: {pred_counts}")
    print(f"Actual labels: {label_counts}")

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
                'dropout': dropout,
                'positive_weight': positive_weight
            }
        }, model_path)

        # Save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(word_to_idx, f)

        print(f"\nBest model saved to {model_path}")
        print(f"Vocabulary saved to {vocab_path}")
        print(
            f"Best validation metrics: Loss={best_val_metrics['loss']:.4f}, Macro F1={best_val_metrics['macro_f1']:.4f} (Epoch {best_val_metrics['epoch']})")

    # Print results
    print(f"\n=== ContextLSTM Results for {task.upper()} Task ===")
    print(f"Power features: {'Yes' if use_power else 'No'}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Binary/Lie F1: {test_metrics['binary_f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")

    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, digits=4, target_names=['Truthful', 'Deceptive']))

    return test_metrics


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