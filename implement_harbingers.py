import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import jsonlines
import os
import re
import argparse
from collections import defaultdict
import joblib
from tqdm import tqdm

# Set paths
project_dir = ""
data_dir = os.path.join(project_dir, "dataset")
test_path = os.path.join(data_dir, "test.jsonl")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")
lexicon_path = os.path.join(project_dir, "utils", "2015_Diplomacy_lexicon.json")

# Create models directory if it doesn't exist
models_dir = os.path.join(project_dir, "models")
os.makedirs(models_dir, exist_ok=True)

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


def load_lexicon():
    """Load the lexicon from the JSON file"""
    try:
        with open(lexicon_path) as f:
            feature_dict = json.loads(f.readline())
        
        # Add additional features as in the original code
        feature_dict['but'] = ['but']
        feature_dict['countries'] = ['austria', 'england', 'france', 'germany', 'italy', 'russia', 'turkey']
        
        return feature_dict
    except FileNotFoundError:
        print(f"Warning: Lexicon file not found at {lexicon_path}")
        print("Using default lexicon instead...")
        
        # Fallback to a simplified lexicon if file not found
        return {
            'planning': ['plan', 'strategy', 'move', 'attack', 'defend', 'support'],
            'certainty': ['absolutely', 'definitely', 'certainly', 'surely'],
            'negation': ['not', 'no', "n't", 'never', 'none'],
            'positive_sentiment': ['good', 'great', 'excellent', 'wonderful'],
            'alliances': ['alliance', 'ally', 'friend', 'partner'],
            'promises': ['promise', 'will', 'shall', 'guarantee'],
            'hedges': ['maybe', 'perhaps', 'possibly', 'probably'],
            'first_person': ['i', 'me', 'my', 'mine', 'we', 'us', 'our'],
            'second_person': ['you', 'your', 'yours', 'yourself'],
            'countries': ['austria', 'england', 'france', 'germany', 'italy', 'russia', 'turkey'],
            'but': ['but']
        }


def extract_features(message_data, task="sender", use_power=True):
    """Extract linguistic harbinger features from messages"""
    features = []
    labels = []
    
    # Load the lexicon
    lexicon = load_lexicon()
    
    # Add progress bar for feature extraction
    for message in tqdm(message_data, desc="Extracting features"):
        # Skip messages without receiver annotation if we're doing the receiver task
        if task.lower() == "receiver" and message['receiver_annotation'] not in [True, False]:
            continue
            
        # Extract the message text and convert to lowercase
        text = message['message'].lower()
        
        # Initialize feature vector for this message
        message_features = []
        
        # Extract linguistic features based on the lexicon
        for category, word_list in lexicon.items():
            # Count how many words from this category appear in the message
            count = 0
            for word in word_list:
                # Use word boundary regex to match whole words only
                count += len(re.findall(r'\b' + re.escape(word) + r'\b', text))
            
            # Add the count as a feature
            message_features.append(count)
        
        # Add message length as a feature
        message_features.append(len(text))
        
        # Add power features if requested
        if use_power:
            # Add raw score delta
            message_features.append(message['score_delta'])
            
            # Add binary power features (severe power imbalance indicators)
            message_features.append(1 if message['score_delta'] > 4 else 0)  # Sender much stronger
            message_features.append(1 if message['score_delta'] < -4 else 0)  # Receiver much stronger
        
        # Add features to the feature list
        features.append(message_features)
        
        # Add the label based on task
        if task.lower() == "sender":
            labels.append(0 if message['sender_annotation'] else 1)  # 0=truthful, 1=deceptive
        else:  # receiver task
            labels.append(0 if message['receiver_annotation'] else 1)  # 0=truthful, 1=deceptive
    
    return np.array(features), np.array(labels)


def run_harbingers_model(task="sender", use_power=True, save_model=True):
    """Run the harbingers model for the specified task"""
    # Load datasets
    print(f"Loading data for {task.upper()} task with power={use_power}")
    with jsonlines.open(train_path, 'r') as reader:
        train_data = list(reader)
    with jsonlines.open(test_path, 'r') as reader:
        test_data = list(reader)
    
    # Process data
    train_messages = aggregate(train_data)
    test_messages = aggregate(test_data)
    
    # Extract features
    X_train, y_train = extract_features(train_messages, task, use_power)
    X_test, y_test = extract_features(test_messages, task, use_power)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model with balanced class weights
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'binary_f1': f1_score(y_test, y_pred, pos_label=1, average='binary'),
        'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    }
    
    # Print results
    print(f"\n=== Harbingers Results for {task.upper()} Task ===")
    print(f"Power features: {'Yes' if use_power else 'No'}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Binary/Lie F1: {metrics['binary_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4, target_names=['Truthful', 'Deceptive']))
    
    # Feature importance
    lexicon = load_lexicon()
    feature_names = list(lexicon.keys()) + ['message_length']
    if use_power:
        feature_names += ['score_delta', 'sender_stronger', 'receiver_stronger']
    
    # Print top positive and negative features
    coef_importance = sorted(zip(feature_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print("\nTop Features for Deception Detection:")
    for name, coef in coef_importance[:5]:
        direction = "increases" if coef > 0 else "decreases"
        print(f"{name}: {coef:.4f} ({direction} likelihood of deception)")
    
    # Save the model and scaler if requested
    if save_model:
        power_suffix = "_with_power" if use_power else "_without_power"
        model_filename = f"harbingers_{task}{power_suffix}.joblib"
        scaler_filename = f"harbingers_scaler_{task}{power_suffix}.joblib"
        
        model_path = os.path.join(models_dir, model_filename)
        scaler_path = os.path.join(models_dir, scaler_filename)
        
        # Save model and scaler
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Harbingers model for deception detection')
    parser.add_argument('--task', choices=['sender', 'receiver'], default='sender',
                        help='Task to perform: "sender" for actual lie detection or "receiver" for suspected lie detection')
    parser.add_argument('--power', action='store_true', help='Use power features')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    args = parser.parse_args()
    
    # Run the model
    metrics = run_harbingers_model(task=args.task, use_power=args.power, save_model=not args.no_save)
    
    # Compare with paper results
    paper_results = {
        'sender': {
            'with_power': {'macro_f1': 0.529, 'binary_f1': 0.237},
            'without_power': {'macro_f1': 0.528, 'binary_f1': 0.246},
        },
        'receiver': {
            'with_power': {'macro_f1': 0.451, 'binary_f1': 0.155},
            'without_power': {'macro_f1': 0.459, 'binary_f1': 0.147},
        }
    }
    
    power_key = 'with_power' if args.power else 'without_power'
    paper_macro_f1 = paper_results[args.task][power_key]['macro_f1']
    paper_binary_f1 = paper_results[args.task][power_key]['binary_f1']
    
    print("\n=== Comparison with Paper Results ===")
    print(f"Our Macro F1: {metrics['macro_f1']:.4f}   Paper: {paper_macro_f1:.4f}")
    print(f"Our Lie F1: {metrics['binary_f1']:.4f}   Paper: {paper_binary_f1:.4f}")
