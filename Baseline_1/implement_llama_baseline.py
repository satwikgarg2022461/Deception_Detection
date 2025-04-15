#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a LLaMA-based baseline for deception detection in Diplomacy dataset
"""

import os
import json
import argparse
import time
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import torch
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Set paths
project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, "dataset")
train_path = os.path.join(data_dir, "train.jsonl")
val_path = os.path.join(data_dir, "validation.jsonl")
test_path = os.path.join(data_dir, "test.jsonl")

# Try to import direct LLaMA inference if available locally
try:
    from llama_cpp import Llama
    HAVE_LLAMA_CPP = True
except ImportError:
    LLAMA_CPP = False
    print("llama-cpp-python not installed. Will use API-based inference.")

# LLaMA model configuration
DEFAULT_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with local path if using llama-cpp
API_BASE_URL = "http://localhost:8080/v1"  # For local API server
TEMPERATURE = 0.05
MAX_TOKENS = 100
DEFAULT_SAMPLE_SIZE = 100  # Number of test samples for evaluation

def process_conversation(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract individual messages from a conversation with their associated metadata
    
    Args:
        conversation: A dictionary containing conversation data
        
    Returns:
        List of individual message dictionaries with metadata
    """
    messages = []
    for i in range(len(conversation['messages'])):
        message = {
            'message_text': conversation['messages'][i],
            'sender': conversation['speakers'][i],
            'receiver': conversation['receivers'][i],
            'sender_label': conversation['sender_labels'][i],  # True = truthful, False = deceptive
            'receiver_label': conversation['receiver_labels'][i],  # True/False/NOANNOTATION
            'game_score': conversation['game_score'][i] if i < len(conversation['game_score']) else None,
            'score_delta': conversation['score_delta'][i] if i < len(conversation['score_delta']) else None,
            'season': conversation['seasons'][i] if i < len(conversation['seasons']) else None,
            'year': conversation['years'][i] if i < len(conversation['years']) else None,
            'game_id': conversation['game_id'],
            'message_idx': i,
            'absolute_msg_idx': conversation['absolute_message_index'][i] if 'absolute_message_index' in conversation else i,
            'relative_msg_idx': conversation['relative_message_index'][i] if 'relative_message_index' in conversation else i,
        }
        messages.append(message)
    return messages

def aggregate_data(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate all messages from all conversations
    
    Args:
        conversations: List of conversation dictionaries
        
    Returns:
        List of all messages with their metadata
    """
    all_messages = []
    for conversation in conversations:
        messages = process_conversation(conversation)
        all_messages.extend(messages)
    return all_messages

def get_conversation_context(data: List[Dict[str, Any]], game_id: str, 
                             msg_idx: int, window_size: int = 5) -> List[Dict[str, Any]]:
    """
    Get previous messages as context for a given message
    
    Args:
        data: List of all messages
        game_id: Game ID to filter by
        msg_idx: Relative message index to get context for
        window_size: Number of previous messages to include as context
        
    Returns:
        List of previous message dictionaries as context
    """
    context = []
    # Filter messages from the same game
    game_messages = [m for m in data if m['game_id'] == game_id]
    # Sort by relative message index to ensure correct order
    game_messages.sort(key=lambda x: x['relative_msg_idx'])
    
    # Get current message
    current_msg = None
    for msg in game_messages:
        if msg['relative_msg_idx'] == msg_idx:
            current_msg = msg
            break
    
    if not current_msg:
        return []
        
    # Get previous messages
    start_idx = max(0, game_messages.index(current_msg) - window_size)
    context_messages = game_messages[start_idx:game_messages.index(current_msg)]
    
    return context_messages

def create_prompt(message: Dict[str, Any], context: List[Dict[str, Any]], 
                  use_power: bool = False) -> str:
    """
    Create a prompt for the LLM
    
    Args:
        message: Dictionary containing the message and metadata
        context: List of previous messages as context
        use_power: Whether to include power features (game scores)
        
    Returns:
        Formatted prompt string for the LLM
    """
    system_prompt = """You are an expert deception detection system analyzing messages from the game of Diplomacy. 
Your task is to determine if a message from one player to another is truthful or deceptive.
Respond with ONLY "truthful" or "deceptive".

In Diplomacy, deception happens when:
1. A player is deliberately lying about their intentions
2. A player is making false promises or misleading statements
3. A player is trying to manipulate others through dishonest means

Analysis should focus on linguistic cues, strategic context, and the power dynamics between players.
"""

    # Format context messages
    context_text = ""
    if context:
        context_text = "Previous messages in this conversation:\n"
        for idx, msg in enumerate(context):
            context_text += f"[{idx+1}] From {msg['sender']} to {msg['receiver']}: {msg['message_text']}\n"
        context_text += "\n"
    
    # Format power information if requested
    power_text = ""
    if use_power and message.get('score_delta') is not None:
        score_delta = message.get('score_delta')
        power_text = f"\nPower difference between sender and receiver (positive means sender has more power): {score_delta}"
    
    # Format current message
    sender = message['sender']
    receiver = message['receiver']
    msg_text = message['message_text']
    
    # Combine all parts
    prompt = f"{system_prompt}\n\n{context_text}Current message from {sender} to {receiver}:\n\"{msg_text}\"{power_text}\n\nIs this message truthful or deceptive?"
    
    return prompt

def call_llama_api(prompt: str, api_url: str = API_BASE_URL) -> str:
    """
    Call LLaMA API to get a response
    
    Args:
        prompt: The formatted prompt to send to the API
        api_url: Base URL for the API
        
    Returns:
        Response text from the API
    """
    headers = {
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response_json = response.json()
        result = response_json["choices"][0]["message"]["content"].strip().lower()
        
        # Extract just the prediction
        if "truthful" in result:
            return "truthful"
        elif "deceptive" in result:
            return "deceptive"
        else:
            # Default to truthful if unclear (more common in dataset)
            return "truthful"
    except Exception as e:
        print(f"Error calling LLaMA API: {e}")
        # Default to truthful if API fails
        return "truthful"

def call_local_llama(prompt: str, model_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Call local LLaMA model using llama-cpp-python
    
    Args:
        prompt: The formatted prompt
        model_path: Path to the LLaMA model file
        
    Returns:
        Response text from the model
    """
    if not HAVE_LLAMA_CPP:
        raise ImportError("llama-cpp-python is not installed")
    
    model = Llama(model_path=model_path)
    response = model(
        prompt, 
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["</s>", "\n\n"],
        echo=False
    )
    
    result = response["choices"][0]["text"].strip().lower()
    
    # Extract just the prediction
    if "truthful" in result:
        return "truthful"
    elif "deceptive" in result:
        return "deceptive"
    else:
        # Default to truthful if unclear (more common in dataset)
        return "truthful"

def predict_message(message: Dict[str, Any], data: List[Dict[str, Any]],
                  use_power: bool = False, local_model: bool = False,
                  model_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Predict truthfulness of a message using LLaMA
    
    Args:
        message: Message dictionary to predict
        data: List of all messages for context
        use_power: Whether to include power features
        local_model: Whether to use local model or API
        model_path: Path to local model file
        
    Returns:
        Prediction ("truthful" or "deceptive")
    """
    # Get conversation context
    context = get_conversation_context(
        data, 
        message['game_id'], 
        message['relative_msg_idx']
    )
    
    # Create the prompt
    prompt = create_prompt(message, context, use_power)
    
    # Get prediction from model
    if local_model and HAVE_LLAMA_CPP:
        return call_local_llama(prompt, model_path)
    else:
        return call_llama_api(prompt)

def run_evaluation(task: str = "sender", use_power: bool = False,
                 sample_size: int = DEFAULT_SAMPLE_SIZE, 
                 local_model: bool = False,
                 model_path: str = DEFAULT_MODEL_PATH,
                 seed: int = 42) -> Dict[str, float]:
    """
    Run evaluation on test set
    
    Args:
        task: "sender" for actual lie detection or "receiver" for suspected lie detection
        use_power: Whether to include power features
        sample_size: Number of test samples to evaluate
        local_model: Whether to use local model or API
        model_path: Path to local model file
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Running LLaMA 3.1-8B-Instruct baseline for {task.upper()} task "
          f"(power={use_power}, sample_size={sample_size})")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Load test data
    with jsonlines.open(test_path, 'r') as reader:
        test_data = list(reader)
    
    # Process data
    test_messages = aggregate_data(test_data)
    
    # Filter messages based on task
    if task.lower() == "receiver":
        # For receiver task, filter out messages without receiver annotations
        test_messages = [msg for msg in test_messages if msg['receiver_label'] != 'NOANNOTATION']
    
    # Select sample for evaluation
    if sample_size < len(test_messages):
        indices = np.random.choice(len(test_messages), sample_size, replace=False)
        test_sample = [test_messages[i] for i in indices]
    else:
        test_sample = test_messages
        print(f"Using all {len(test_sample)} available test messages")
    
    # Run predictions
    predictions = []
    true_labels = []
    
    for message in tqdm(test_sample, desc="Evaluating with LLaMA"):
        # Get ground truth label based on task
        if task.lower() == "sender":
            # True = truthful (0), False = deceptive (1)
            true_label = 0 if message['sender_label'] else 1
        else:  # receiver task
            true_label = 0 if message['receiver_label'] else 1
        
        # Get prediction
        result = predict_message(
            message, 
            test_messages, 
            use_power=use_power,
            local_model=local_model,
            model_path=model_path
        )
        
        # Convert prediction to numerical label
        pred_label = 0 if result == "truthful" else 1
        
        predictions.append(pred_label)
        true_labels.append(true_label)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'macro_f1': f1_score(true_labels, predictions, average='macro'),
        'lie_f1': f1_score(true_labels, predictions, average='binary', pos_label=1),
        'precision': precision_score(true_labels, predictions, pos_label=1, zero_division=0),
        'recall': recall_score(true_labels, predictions, pos_label=1, zero_division=0)
    }
    
    # Print results
    print(f"\n=== LLaMA 3.1-8B-Instruct Results for {task.upper()} Task ===")
    print(f"Power features: {'Yes' if use_power else 'No'}")
    print(f"Sample size: {len(test_sample)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Lie F1: {metrics['lie_f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions, digits=4, target_names=['Truthful', 'Deceptive']))
    
    # Compare with paper results
    paper_results = {
        'sender': {
            'Human': {'macro_f1': 0.579, 'lie_f1': 0.480},
            'BERT+Context': {'macro_f1': 0.672, 'lie_f1': 0.471},
            'LSTM': {'macro_f1': 0.622, 'lie_f1': 0.394},
            'Bag of Words': {'macro_f1': 0.536, 'lie_f1': 0.340},
            'Random': {'macro_f1': 0.494, 'lie_f1': 0.298},
            'Majority Class': {'macro_f1': 0.413, 'lie_f1': 0.000}
        },
        'receiver': {
            'BERT+Context': {'macro_f1': 0.732, 'lie_f1': 0.630},
            'LSTM': {'macro_f1': 0.678, 'lie_f1': 0.562},
            'Bag of Words': {'macro_f1': 0.608, 'lie_f1': 0.441},
            'Random': {'macro_f1': 0.493, 'lie_f1': 0.404},
            'Majority Class': {'macro_f1': 0.442, 'lie_f1': 0.000}
        }
    }
    
    print("\n=== Comparison with Paper Results ===")
    task_key = task.lower()
    
    if task_key == "sender":
        print(f"Human (Paper): Macro F1={paper_results[task_key]['Human']['macro_f1']:.4f}, "
              f"Lie F1={paper_results[task_key]['Human']['lie_f1']:.4f}")
              
    print(f"BERT+Context (Paper): Macro F1={paper_results[task_key]['BERT+Context']['macro_f1']:.4f}, "
          f"Lie F1={paper_results[task_key]['BERT+Context']['lie_f1']:.4f}")
          
    print(f"LSTM (Paper): Macro F1={paper_results[task_key]['LSTM']['macro_f1']:.4f}, "
          f"Lie F1={paper_results[task_key]['LSTM']['lie_f1']:.4f}")
          
    print(f"Bag of Words (Paper): Macro F1={paper_results[task_key]['Bag of Words']['macro_f1']:.4f}, "
          f"Lie F1={paper_results[task_key]['Bag of Words']['lie_f1']:.4f}")
          
    print(f"Random (Paper): Macro F1={paper_results[task_key]['Random']['macro_f1']:.4f}, "
          f"Lie F1={paper_results[task_key]['Random']['lie_f1']:.4f}")
          
    print(f"LLaMA 3.1-8B-Instruct (Ours): Macro F1={metrics['macro_f1']:.4f}, "
          f"Lie F1={metrics['lie_f1']:.4f}")
    
    # Save results to file
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(
        results_dir, 
        f"llama_results_{task}_{use_power}_{timestamp}.json"
    )
    
    with open(results_file, 'w') as f:
        json.dump({
            'task': task,
            'use_power': use_power,
            'sample_size': len(test_sample),
            'metrics': metrics,
            'config': {
                'model': 'LLaMA-3.1-8B-Instruct',
                'temperature': TEMPERATURE,
                'max_tokens': MAX_TOKENS,
                'local_model': local_model
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return metrics

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run LLaMA baseline for deception detection')
    parser.add_argument('--task', choices=['sender', 'receiver'], default='sender',
                        help='Task to perform: "sender" for actual lie detection or "receiver" for suspected lie detection')
    parser.add_argument('--power', action='store_true', help='Use power features')
    parser.add_argument('--sample', type=int, default=DEFAULT_SAMPLE_SIZE, help='Number of test samples to evaluate')
    parser.add_argument('--local', action='store_true', help='Use local model instead of API')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to local model file')
    parser.add_argument('--full-eval', action='store_true', help='Run full evaluation (both tasks, with and without power)')
    
    args = parser.parse_args()
    
    if args.full_eval:
        # Run all combinations
        results = {}
        for task in ['sender', 'receiver']:
            for power in [False, True]:
                print(f"\n{'='*80}")
                print(f"Running evaluation: task={task}, power={power}")
                print(f"{'='*80}")
                metrics = run_evaluation(
                    task=task,
                    use_power=power,
                    sample_size=args.sample,
                    local_model=args.local,
                    model_path=args.model
                )
                results[f"{task}_{power}"] = metrics
        
        # Print summary
        print("\n\nSummary of Results:")
        print(f"{'='*80}")
        for key, metrics in results.items():
            task, power = key.split('_')
            print(f"{task.capitalize()} task, power={power}: "
                  f"Accuracy={metrics['accuracy']:.4f}, "
                  f"Macro F1={metrics['macro_f1']:.4f}, "
                  f"Lie F1={metrics['lie_f1']:.4f}")
    else:
        # Run single evaluation
        run_evaluation(
            task=args.task,
            use_power=args.power,
            sample_size=args.sample,
            local_model=args.local,
            model_path=args.model
        )

if __name__ == "__main__":
    main()