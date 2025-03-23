import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import jsonlines
import os
import argparse
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from scipy.sparse import csr_matrix, hstack
import joblib
from tqdm import tqdm

# Set paths
project_dir = "d:\\NLP\\Deception_Detection"
data_dir = os.path.join(project_dir, "dataset")
test_path = os.path.join(data_dir, "test.jsonl")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")

# Create models directory if it doesn't exist
models_dir = os.path.join(project_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# Initialize spaCy
nlp = English()

# Define tokenizer function (same as the original)
def is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False

def spacy_tokenizer(text):
    return [tok.text if not is_number(tok.text) else '_NUM_' for tok in nlp(text)]

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

def run_bow_model(task="sender", use_power=True, save_model=True):
    """Run the Bag of Words model for the specified task"""
    # Load datasets
    print(f"Loading data for {task.upper()} task with power={use_power}")
    with jsonlines.open(train_path, 'r') as reader:
        train_data = list(reader)
    with jsonlines.open(test_path, 'r') as reader:
        test_data = list(reader)
    
    # Process data
    print("Processing data...")
    train_messages = aggregate(train_data)
    test_messages = aggregate(test_data)
    
    # Create vectorizer with spaCy tokenizer
    vectorizer = CountVectorizer(
        tokenizer=spacy_tokenizer,
        stop_words=list(STOP_WORDS),
        strip_accents='unicode'
    )
    
    # Prepare corpus based on task
    print("Creating document corpus...")
    if task.lower() == "sender":
        train_corpus = [message['message'].lower() for message in tqdm(train_messages, desc="Processing train messages")]
        test_corpus = [message['message'].lower() for message in tqdm(test_messages, desc="Processing test messages")]
        
        y_train = [0 if message['sender_annotation'] else 1 for message in train_messages]
        y_test = [0 if message['sender_annotation'] else 1 for message in test_messages]
    else:  # receiver task - filter out messages without annotations
        train_corpus = [message['message'].lower() for message in train_messages 
                       if message['receiver_annotation'] in [True, False]]
        test_corpus = [message['message'].lower() for message in test_messages 
                      if message['receiver_annotation'] in [True, False]]
        
        y_train = [0 if message['receiver_annotation'] else 1 
                  for message in train_messages if message['receiver_annotation'] in [True, False]]
        y_test = [0 if message['receiver_annotation'] else 1 
                 for message in test_messages if message['receiver_annotation'] in [True, False]]
    
    print("Vectorizing text...")
    # Transform corpus to BOW features
    X_train_bow = vectorizer.fit_transform(train_corpus)
    X_test_bow = vectorizer.transform(test_corpus)
    
    # Add power features if requested
    if use_power:
        # Prepare power features for training data
        if task.lower() == "sender":
            train_power_features = []
            for message in train_messages:
                # Binary power features (same as original)
                power_feature = [
                    1 if message['score_delta'] > 4 else 0,  # Sender much stronger
                    1 if message['score_delta'] < -4 else 0   # Receiver much stronger
                ]
                train_power_features.append(power_feature)
            
            test_power_features = []
            for message in test_messages:
                power_feature = [
                    1 if message['score_delta'] > 4 else 0,
                    1 if message['score_delta'] < -4 else 0
                ]
                test_power_features.append(power_feature)
        else:  # receiver task - only for annotated messages
            train_power_features = []
            test_power_features = []
            
            for message in train_messages:
                if message['receiver_annotation'] in [True, False]:
                    power_feature = [
                        1 if message['score_delta'] > 4 else 0,
                        1 if message['score_delta'] < -4 else 0
                    ]
                    train_power_features.append(power_feature)
            
            for message in test_messages:
                if message['receiver_annotation'] in [True, False]:
                    power_feature = [
                        1 if message['score_delta'] > 4 else 0,
                        1 if message['score_delta'] < -4 else 0
                    ]
                    test_power_features.append(power_feature)
        
        # Convert to numpy arrays
        train_power_array = np.array(train_power_features)
        test_power_array = np.array(test_power_features)
        
        # Combine BOW features with power features (same as original)
        X_train = hstack([X_train_bow, csr_matrix(train_power_array)])
        X_test = hstack([X_test_bow, csr_matrix(test_power_array)])
    else:
        # Use only BOW features
        X_train = X_train_bow
        X_test = X_test_bow
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Train logistic regression model with balanced class weights (same as original)
    print("Training model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'binary_f1': f1_score(y_test, y_pred, pos_label=1, average='binary'),
        'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    }
    
    # Print results
    print(f"\n=== Bag of Words Results for {task.upper()} Task ===")
    print(f"Power features: {'Yes' if use_power else 'No'}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Binary/Lie F1: {metrics['binary_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4, target_names=['Truthful', 'Deceptive']))
    
    # Display some of the most influential words (similar to original)
    if hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
        top_positive_coefs = np.argsort(model.coef_[0])[-10:]
        top_negative_coefs = np.argsort(model.coef_[0])[:10]
        
        print("\nTop words indicating deception:")
        for idx in top_positive_coefs:
            if idx < len(feature_names):  # Ensure we're not accessing power features
                print(f"{feature_names[idx]}: {model.coef_[0][idx]:.4f}")
        
        print("\nTop words indicating truth:")
        for idx in top_negative_coefs:
            if idx < len(feature_names):  # Ensure we're not accessing power features
                print(f"{feature_names[idx]}: {model.coef_[0][idx]:.4f}")
    
    # Save the model and vectorizer if requested
    if save_model:
        power_suffix = "_with_power" if use_power else "_without_power"
        model_filename = f"bow_{task}{power_suffix}.joblib"
        vectorizer_filename = f"bow_vectorizer_{task}{power_suffix}.joblib"
        
        model_path = os.path.join(models_dir, model_filename)
        vectorizer_path = os.path.join(models_dir, vectorizer_filename)
        
        # Save model and vectorizer
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Bag of Words model for deception detection')
    parser.add_argument('--task', choices=['sender', 'receiver'], default='sender',
                        help='Task to perform: "sender" for actual lie detection or "receiver" for suspected lie detection')
    parser.add_argument('--power', action='store_true', help='Use power features')
    parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    args = parser.parse_args()
    
    # Run the model
    metrics = run_bow_model(task=args.task, use_power=args.power, save_model=not args.no_save)
    
    # Compare with paper results
    paper_results = {
        'sender': {
            'with_power': {'macro_f1': 0.539, 'binary_f1': 0.383},
            'without_power': {'macro_f1': 0.536, 'binary_f1': 0.340},
        },
        'receiver': {
            'with_power': {'macro_f1': 0.611, 'binary_f1': 0.469},
            'without_power': {'macro_f1': 0.608, 'binary_f1': 0.441},
        }
    }
    
    power_key = 'with_power' if args.power else 'without_power'
    paper_macro_f1 = paper_results[args.task][power_key]['macro_f1']
    paper_binary_f1 = paper_results[args.task][power_key]['binary_f1']
    
    print("\n=== Comparison with Paper Results ===")
    print(f"Our Macro F1: {metrics['macro_f1']:.4f}   Paper: {paper_macro_f1:.4f}")
    print(f"Our Binary F1: {metrics['binary_f1']:.4f}   Paper: {paper_binary_f1:.4f}")
