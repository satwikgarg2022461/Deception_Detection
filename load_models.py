import os
import torch
import joblib
import pickle
import numpy as np

project_dir = ""
models_dir = os.path.join(project_dir, "models")

def load_harbingers_model(task="sender", use_power=True):
    """Load a saved Harbingers model"""
    power_suffix = "_with_power" if use_power else "_without_power"
    model_filename = f"harbingers_{task}{power_suffix}.joblib"
    scaler_filename = f"harbingers_scaler_{task}{power_suffix}.joblib"
    
    model_path = os.path.join(models_dir, model_filename)
    scaler_path = os.path.join(models_dir, scaler_filename)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model files not found at {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def load_bow_model(task="sender", use_power=True):
    """Load a saved Bag of Words model"""
    power_suffix = "_with_power" if use_power else "_without_power"
    model_filename = f"bow_{task}{power_suffix}.joblib"
    vectorizer_filename = f"bow_vectorizer_{task}{power_suffix}.joblib"
    
    model_path = os.path.join(models_dir, model_filename)
    vectorizer_path = os.path.join(models_dir, vectorizer_filename)
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(f"Error: Model files not found at {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer

def load_lstm_model(task="sender", use_power=True):
    """Load a saved LSTM model"""
    from implement_lstm import LSTMModel
    
    power_suffix = "_with_power" if use_power else "_without_power"
    model_filename = f"lstm_{task}{power_suffix}.pt"
    vocab_filename = f"lstm_vocab_{task}{power_suffix}.pkl"
    
    model_path = os.path.join(models_dir, model_filename)
    vocab_path = os.path.join(models_dir, vocab_filename)
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"Error: Model files not found at {model_path}")
        return None, None
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        word_to_idx = pickle.load(f)
    
    # Load model configuration and weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create model with saved configuration
    model = LSTMModel(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=2,
        use_power=checkpoint['config']['use_power'],
        dropout=checkpoint['config']['dropout']
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, word_to_idx

def load_contextlstm_model(task="sender", use_power=True):
    """Load a saved ContextLSTM model"""
    from implement_contextlstm import ContextLSTMModel
    
    power_suffix = "_with_power" if use_power else "_without_power"
    model_filename = f"contextlstm_{task}{power_suffix}.pt"
    vocab_filename = f"contextlstm_vocab_{task}{power_suffix}.pkl"
    
    model_path = os.path.join(models_dir, model_filename)
    vocab_path = os.path.join(models_dir, vocab_filename)
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"Error: Model files not found at {model_path}")
        return None, None
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        word_to_idx = pickle.load(f)
    
    # Load model configuration and weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create model with saved configuration
    model = ContextLSTMModel(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        message_hidden_dim=checkpoint['message_hidden_dim'],
        conv_hidden_dim=checkpoint['conv_hidden_dim'],
        output_dim=2,
        use_power=checkpoint['config']['use_power'],
        dropout=checkpoint['dropout']
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, word_to_idx

def load_bertcontext_model(task="sender", use_power=True):
    """Load a saved BERT+Context model"""
    from implement_bertcontext import BERTContextModel
    
    power_suffix = "_with_power" if use_power else "_without_power"
    model_filename = f"bertcontext_{task}{power_suffix}.pt"
    
    model_path = os.path.join(models_dir, model_filename)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    # Load model configuration and weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create model with saved configuration
    model = BERTContextModel(
        conv_hidden_dim=checkpoint['conv_hidden_dim'],
        output_dim=2,
        use_power=checkpoint['config']['use_power'],
        dropout=checkpoint['dropout']
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

if __name__ == "__main__":
    print("Available models:")
    for filename in sorted(os.listdir(models_dir)):
        print(f"  - {filename}")
