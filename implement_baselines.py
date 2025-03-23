import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from random import uniform
import jsonlines
import os

# Set paths
project_dir = ""
data_dir = os.path.join(project_dir, "dataset")
test_path = os.path.join(data_dir, "test.jsonl")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")

# Number of repeats for averaging random baseline results
repeats = 500

# Convert conversations into single messages (same as original)
def aggregate(dataset):
    messages = []
    rec = []
    send = []
    for dialogs in dataset:
        messages.extend(dialogs['messages'])
        rec.extend(dialogs['receiver_labels'])
        send.extend(dialogs['sender_labels'])
    merged = []
    for i, item in enumerate(messages):
        merged.append({'message': item, 'sender_annotation': send[i], 'receiver_annotation': rec[i]})
    return merged

def extract_labels(dataset):
    """Extract sender and receiver labels from the dataset"""
    sender_labels = []
    receiver_labels = []
    
    for msg in aggregate(dataset):
        # Extract sender labels (actual lies) - don't drop any
        if msg['sender_annotation'] == True:
            sender_labels.append(0)  # 0 = truthful
        elif msg['sender_annotation'] == False:
            sender_labels.append(1)  # 1 = deceptive
            
        # Extract receiver labels (suspected lies) - only include annotated ones
        if msg['receiver_annotation'] == True:
            receiver_labels.append(0)  # 0 = perceived truthful
        elif msg['receiver_annotation'] == False:
            receiver_labels.append(1)  # 1 = perceived deceptive
            
    return sender_labels, receiver_labels

def run_random_baseline(labels, repeats=500):
    """Run random baseline multiple times and return average metrics"""
    random_preds_all = []
    metrics = {
        'macro_f1': [], 
        'binary_f1': [], 
        'accuracy': [],
        'precision': [],
        'recall': []
    }
    
    # Add progress bar for iterations
    from tqdm import tqdm
    for _ in tqdm(range(repeats), desc="Running random baseline"):
        # Generate random predictions (coin flip)
        random_preds = [1 if uniform(0, 1) > 0.5 else 0 for _ in range(len(labels))]
        random_preds_all.append(random_preds)
        
        # Calculate metrics
        metrics['macro_f1'].append(f1_score(labels, random_preds, pos_label=1, average='macro'))
        metrics['binary_f1'].append(f1_score(labels, random_preds, pos_label=1, average='binary'))
        metrics['accuracy'].append(accuracy_score(labels, random_preds))
        metrics['precision'].append(precision_score(labels, random_preds, pos_label=1, zero_division=0))
        metrics['recall'].append(recall_score(labels, random_preds, pos_label=1, zero_division=0))
    
    # Average the metrics
    for metric in metrics:
        metrics[metric] = sum(metrics[metric]) / repeats
        
    return metrics

def run_majority_baseline(labels):
    """Run majority class baseline and return metrics"""
    # Find the majority class
    unique_classes, counts = np.unique(labels, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]
    
    # Generate majority class predictions
    majority_preds = [majority_class] * len(labels)
    
    # Calculate metrics
    metrics = {
        'macro_f1': f1_score(labels, majority_preds, pos_label=1, average='macro'),
        'binary_f1': f1_score(labels, majority_preds, pos_label=1, average='binary'),
        'accuracy': accuracy_score(labels, majority_preds),
        'precision': precision_score(labels, majority_preds, pos_label=1, zero_division=0),
        'recall': recall_score(labels, majority_preds, pos_label=1, zero_division=0)
    }
    
    return metrics, majority_class

if __name__ == '__main__':
    # Load test data
    print("Loading test data from:", test_path)
    with jsonlines.open(test_path, 'r') as reader:
        test_data = list(reader)
        
    # Extract labels
    sender_labels, receiver_labels = extract_labels(test_data)
    print("Total sender samples:", len(sender_labels))
    print("Total receiver samples:", len(receiver_labels))
    
    # Calculate class distribution
    sender_pos_rate = sum(sender_labels) / len(sender_labels)
    receiver_pos_rate = sum(receiver_labels) / len(receiver_labels)
    print(f"Sender positive rate (lies): {sender_pos_rate:.4f} ({sender_pos_rate*100:.2f}%)")
    print(f"Receiver positive rate (perceived lies): {receiver_pos_rate:.4f} ({receiver_pos_rate*100:.2f}%)")
    
    # Run random baseline
    print("\n=== Random Baseline ===")
    sender_random_metrics = run_random_baseline(sender_labels, repeats)
    receiver_random_metrics = run_random_baseline(receiver_labels, repeats)
    
    print("Sender (Actual Lie) Random Baseline:")
    print(f"  Macro F1: {sender_random_metrics['macro_f1']:.4f}")
    print(f"  Binary/Lie F1: {sender_random_metrics['binary_f1']:.4f}")
    print(f"  Accuracy: {sender_random_metrics['accuracy']:.4f}")
    
    print("Receiver (Suspected Lie) Random Baseline:")
    print(f"  Macro F1: {receiver_random_metrics['macro_f1']:.4f}")
    print(f"  Binary/Lie F1: {receiver_random_metrics['binary_f1']:.4f}")
    print(f"  Accuracy: {receiver_random_metrics['accuracy']:.4f}")
    
    # Run majority baseline
    print("\n=== Majority Class Baseline ===")
    sender_majority_metrics, sender_majority_class = run_majority_baseline(sender_labels)
    receiver_majority_metrics, receiver_majority_class = run_majority_baseline(receiver_labels)
    
    print(f"Sender majority class: {sender_majority_class} ({'truthful' if sender_majority_class == 0 else 'deceptive'})")
    print("Sender (Actual Lie) Majority Baseline:")
    print(f"  Macro F1: {sender_majority_metrics['macro_f1']:.4f}")
    print(f"  Binary/Lie F1: {sender_majority_metrics['binary_f1']:.4f}")
    print(f"  Accuracy: {sender_majority_metrics['accuracy']:.4f}")
    
    print(f"Receiver majority class: {receiver_majority_class} ({'truthful' if receiver_majority_class == 0 else 'deceptive'})")
    print("Receiver (Suspected Lie) Majority Baseline:")
    print(f"  Macro F1: {receiver_majority_metrics['macro_f1']:.4f}")
    print(f"  Binary/Lie F1: {receiver_majority_metrics['binary_f1']:.4f}")
    print(f"  Accuracy: {receiver_majority_metrics['accuracy']:.4f}")
    
    # Compare with paper results
    print("\n=== Comparison with Paper Results ===")
    print("                      | Our Implementation | Paper Results")
    print("---------------------------------------------------------")
    print(f"Sender Random Macro F1  | {sender_random_metrics['macro_f1']:.4f}              | 0.398")
    print(f"Sender Random Lie F1    | {sender_random_metrics['binary_f1']:.4f}              | 0.149")
    print(f"Sender Majority Macro F1| {sender_majority_metrics['macro_f1']:.4f}              | 0.478")
    print(f"Sender Majority Lie F1  | {sender_majority_metrics['binary_f1']:.4f}              | 0.000")
    print(f"Receiver Random Macro F1| {receiver_random_metrics['macro_f1']:.4f}              | 0.383")
    print(f"Receiver Random Lie F1  | {receiver_random_metrics['binary_f1']:.4f}              | 0.118")
    print(f"Receiver Majority Mac F1| {receiver_majority_metrics['macro_f1']:.4f}              | 0.483")
    print(f"Receiver Majority Lie F1| {receiver_majority_metrics['binary_f1']:.4f}              | 0.000")
