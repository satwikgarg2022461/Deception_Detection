import json
import os
import time
# Use the standard OpenAI library, but point it to the local server
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import re # Import regular expressions for more robust parsing

# --- Configuration ---

# !! EDIT THESE !!
LOCAL_LLM_BASE_URL = "http://localhost:11434/v1" # Default for Ollama. Change if your server uses a different port/path.
# LOCAL_LLM_MODEL_NAME = "llama3:instruct" # Replace with the model you downloaded (e.g., "mistral:instruct", "phi3:instruct")
LOCAL_LLM_MODEL_NAME = "llama3:instruct" # Using Mistral as an example
LOCAL_API_KEY = "ollama" # Or "NotNeeded", "placeholder", etc.

# Other configurations
INPUT_FILE = "../dataset/test.jsonl" # Your input jsonlines file path
MAX_MESSAGES_TO_PROCESS = None # Set to a number (e.g., 20) for quick testing, None to process all
REQUEST_DELAY = 0.1 # Seconds to wait between API calls (can be lower for local models)

# --- OpenAI Client Initialization for Local LLM ---
try:
    client = OpenAI(
        base_url=LOCAL_LLM_BASE_URL,
        api_key=LOCAL_API_KEY, # Placeholder API key
    )
    # Optional: Add a quick check to see if the server is reachable
    # client.models.list()
    print(f"Attempting to connect to local LLM server at: {LOCAL_LLM_BASE_URL} with model: {LOCAL_LLM_MODEL_NAME}")
except Exception as e:
    print(f"Error initializing OpenAI client for local server: {e}")
    print("Please ensure your local LLM server (like Ollama) is running and the BASE_URL is correct.")
    exit(1)

# --- Helper Function for LLM Call (Modified Parsing) ---
def get_llm_prediction(prompt: str, task_description: str, max_retries=3, delay=1) -> str | None:
    """
    Sends a prompt to the Local LLM and returns the predicted label ('True' or 'False').
    Includes basic retry logic and more robust output parsing.
    'task_description' is added for clarity in debugging/logging if needed.
    """
    # print(f"--- Sending Prompt for {task_description} ---") # Optional Debug
    # print(prompt)
    # print("------")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LOCAL_LLM_MODEL_NAME,
                messages=[
                     # System prompt effectiveness varies. Adjust if needed.
                    {"role": "system", "content": "You are an analyst of the game Diplomacy. Analyze messages based on the user's specific request. Respond ONLY with the single word 'True' or 'False'. Do not add explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # Keep low for classification
                max_tokens=20, # Allow a bit more room
                # stop=["\n"] # Optional: try adding stop sequences if model adds extra lines
            )
            raw_prediction_text = response.choices[0].message.content.strip()
            # print(f"Raw LLM Response ({task_description}): '{raw_prediction_text}'") # Optional Debug

            # --- Robust Parsing ---
            found_true = re.search(r'\btrue\b', raw_prediction_text, re.IGNORECASE)
            found_false = re.search(r'\bfalse\b', raw_prediction_text, re.IGNORECASE)

            if found_true and not found_false:
                return "True"
            elif found_false and not found_true:
                return "False"
            elif raw_prediction_text.lower() == "true":
                 return "True"
            elif raw_prediction_text.lower() == "false":
                 return "False"
            else:
                print(f"Warning: LLM returned ambiguous text for {task_description}: '{raw_prediction_text}'. Treating as ambiguous (None).")
                return None

        except Exception as e:
            print(f"Error calling local LLM API for {task_description} (attempt {attempt + 1}/{max_retries}): {e}")
            if "connection error" in str(e).lower():
                 print("Connection Error: Is the local LLM server (Ollama, etc.) running?")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt)) # Exponential backoff
            else:
                print(f"Max retries reached for {task_description}. Skipping this prediction.")
                return None
    return None

# --- Main Processing Logic ---
def process_diplomacy_data(filepath: str):
    """
    Reads the jsonlines file, gets LLM predictions for sender and receiver labels,
    and evaluates them using accuracy and F1 scores.
    """
    all_true_sender_labels = []
    all_predicted_sender_labels = []
    all_true_receiver_labels = []
    all_predicted_receiver_labels = []

    processed_message_count = 0
    skipped_receiver_noannotation = 0

    print(f"Starting analysis of {filepath} using local model: {LOCAL_LLM_MODEL_NAME} via {LOCAL_LLM_BASE_URL}...")

    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if MAX_MESSAGES_TO_PROCESS is not None and processed_message_count >= MAX_MESSAGES_TO_PROCESS:
                    print(f"\nReached maximum message limit ({MAX_MESSAGES_TO_PROCESS}). Stopping.")
                    break
                try:
                    game_data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON on line {line_num + 1}: {e}")
                    continue

                game_id = game_data.get('game_id', 'N/A')
                print(f"\nProcessing Game ID: {game_id}")
                messages = game_data.get('messages', [])
                speakers = game_data.get('speakers', [])
                receivers = game_data.get('receivers', [])
                true_sender_labels_raw = game_data.get('sender_labels', [])
                true_receiver_labels_raw = game_data.get('receiver_labels', []) # Get receiver labels

                if not (len(messages) == len(speakers) == len(receivers) == len(true_sender_labels_raw) == len(true_receiver_labels_raw)):
                    print(f"Warning: Mismatched lengths in Game ID {game_id}. Skipping game.")
                    continue

                for i in range(len(messages)):
                    if MAX_MESSAGES_TO_PROCESS is not None and processed_message_count >= MAX_MESSAGES_TO_PROCESS:
                        break

                    current_message = messages[i]
                    speaker = speakers[i]
                    receiver = receivers[i]
                    true_sender_label = true_sender_labels_raw[i] # Ground truth for sender
                    true_receiver_label_raw = true_receiver_labels_raw[i] # Ground truth for receiver

                    print(f"  Analysing message {i+1}/{len(messages)} (Abs Index: {game_data.get('absolute_message_index', [])[i]})")
                    processed_message_count += 1

                    # --- Prepare Context ---
                    previous_message_text = "N/A (First message in dialog)"
                    if i > 0:
                        previous_message_text = messages[i-1]

                    base_context = f"""
Context: Game of Diplomacy.
Sender: {speaker}
Receiver: {receiver}
Previous Message in this dialog: "{previous_message_text}"
Current Message: "{current_message}"
"""

                    # === Sender Label Prediction ===
                    sender_prompt = f"""{base_context}
Task: Predict the sender's label for the *Current Message*.
'True' means the sender seems strategically genuine/cooperative *at this moment*.
'False' means the sender seems deceptive/manipulative *at this moment*.

Respond ONLY with the single word 'True' or 'False'.
"""
                    predicted_sender_label_str = get_llm_prediction(sender_prompt, "Sender Label")

                    if predicted_sender_label_str is not None:
                        predicted_sender_label = predicted_sender_label_str == "True"
                        all_true_sender_labels.append(true_sender_label)
                        all_predicted_sender_labels.append(predicted_sender_label)
                        print(f"    Sender Label -> True: {true_sender_label}, Predicted: {predicted_sender_label} ({predicted_sender_label_str})")
                    else:
                        print(f"    Sender Label -> Prediction failed/invalid. Skipping for this message.")
                        # Skip receiver prediction if sender prediction failed? Optional, but makes sense.
                        if REQUEST_DELAY > 0: time.sleep(REQUEST_DELAY)
                        continue # Skip to next message if sender prediction failed

                    if REQUEST_DELAY > 0: time.sleep(REQUEST_DELAY) # Delay between sender/receiver calls

                    # === Receiver Label Prediction (Only if not 'NOANNOTATION') ===
                    if true_receiver_label_raw == "NOANNOTATION":
                        print(f"    Receiver Label -> Skipping (True label is 'NOANNOTATION')")
                        skipped_receiver_noannotation += 1
                    elif isinstance(true_receiver_label_raw, bool): # Check if it's True/False
                        true_receiver_label = true_receiver_label_raw

                        receiver_prompt = f"""{base_context}
Task: Predict the receiver's *likely perception* of the *Current Message*.
'True' means the receiver likely perceives the message as strategically genuine/cooperative *from the sender*.
'False' means the receiver likely perceives the message as deceptive/manipulative *from the sender*.

Respond ONLY with the single word 'True' or 'False'.
"""
                        predicted_receiver_label_str = get_llm_prediction(receiver_prompt, "Receiver Label")

                        if predicted_receiver_label_str is not None:
                            predicted_receiver_label = predicted_receiver_label_str == "True"
                            all_true_receiver_labels.append(true_receiver_label)
                            all_predicted_receiver_labels.append(predicted_receiver_label)
                            print(f"    Receiver Label -> True: {true_receiver_label}, Predicted: {predicted_receiver_label} ({predicted_receiver_label_str})")
                        else:
                            print(f"    Receiver Label -> Prediction failed/invalid. Skipping receiver label for this message.")
                    else:
                        # Should not happen based on schema, but good to catch
                         print(f"    Receiver Label -> Skipping (Unexpected true label format: {true_receiver_label_raw})")


                    # --- Delay before next message ---
                    if REQUEST_DELAY > 0: time.sleep(REQUEST_DELAY)


    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")


    # --- Evaluation ---
    print(f"\n--- Evaluation Results ---")
    print(f"Total messages processed: {processed_message_count}")

    # --- Sender Label Evaluation ---
    print("\n--- Sender Labels ---")
    if not all_true_sender_labels:
        print("No valid sender label predictions were made. Cannot calculate metrics.")
    else:
        print(f"Number of Sender Labels Evaluated: {len(all_true_sender_labels)}")
        # Accuracy
        sender_accuracy = accuracy_score(all_true_sender_labels, all_predicted_sender_labels)
        print(f"Sender Accuracy: {sender_accuracy:.4f}")

        # F1 Scores
        # Binary F1: F1 score for the positive class (assuming True is positive)
        sender_f1_binary = f1_score(all_true_sender_labels, all_predicted_sender_labels, pos_label=False, average='binary', zero_division=0)
        print(f"Sender F1 Score (Binary, for True): {sender_f1_binary:.4f}")
        # Micro F1: Accuracy, essentially sum of TP / (sum of TP + FP) across classes
        sender_f1_micro = f1_score(all_true_sender_labels, all_predicted_sender_labels, average='micro', zero_division=0)
        print(f"Sender F1 Score (Micro): {sender_f1_micro:.4f}") # Should be same as accuracy for binary

        # Confusion Matrix & Classification Report
        print("\nSender Confusion Matrix:")
        cm_sender = confusion_matrix(all_true_sender_labels, all_predicted_sender_labels, labels=[True, False])
        print("        Predicted True  Predicted False")
        print(f"Actual True    {cm_sender[0][0]:<14} {cm_sender[0][1]:<15}")
        print(f"Actual False   {cm_sender[1][0]:<14} {cm_sender[1][1]:<15}")

        print("\nSender Classification Report:")
        report_sender = classification_report(all_true_sender_labels, all_predicted_sender_labels, target_names=['False', 'True'], zero_division=0)
        print(report_sender)

    # --- Receiver Label Evaluation ---
    print("\n--- Receiver Labels ---")
    print(f"(Skipped {skipped_receiver_noannotation} messages due to 'NOANNOTATION' true label)")
    if not all_true_receiver_labels:
        print("No valid receiver label predictions were made (or all were 'NOANNOTATION'). Cannot calculate metrics.")
    else:
        print(f"Number of Receiver Labels Evaluated: {len(all_true_receiver_labels)}")
        # Accuracy
        receiver_accuracy = accuracy_score(all_true_receiver_labels, all_predicted_receiver_labels)
        print(f"Receiver Accuracy: {receiver_accuracy:.4f}")

        # F1 Scores
        receiver_f1_binary = f1_score(all_true_receiver_labels, all_predicted_receiver_labels, pos_label=False, average='binary', zero_division=0)
        print(f"Receiver F1 Score (Binary, for True): {receiver_f1_binary:.4f}")
        receiver_f1_micro = f1_score(all_true_receiver_labels, all_predicted_receiver_labels, average='micro', zero_division=0)
        print(f"Receiver F1 Score (Micro): {receiver_f1_micro:.4f}") # Should be same as accuracy

        # Confusion Matrix & Classification Report
        print("\nReceiver Confusion Matrix:")
        # Ensure labels=[True, False] covers the possible boolean values
        cm_receiver = confusion_matrix(all_true_receiver_labels, all_predicted_receiver_labels, labels=[True, False])
        print("        Predicted True  Predicted False")
        print(f"Actual True    {cm_receiver[0][0]:<14} {cm_receiver[0][1]:<15}")
        print(f"Actual False   {cm_receiver[1][0]:<14} {cm_receiver[1][1]:<15}")

        print("\nReceiver Classification Report:")
        report_receiver = classification_report(all_true_receiver_labels, all_predicted_receiver_labels, target_names=['False', 'True'], zero_division=0)
        print(report_receiver)


# --- Run the analysis ---
if __name__ == "__main__":
    # Prepare a dummy data file for demonstration if it doesn't exist
    if not os.path.exists(INPUT_FILE):
        print(f"Creating dummy data file: {INPUT_FILE}")
        # Ensure dummy data has varied receiver labels including NOANNOTATION and Booleans
        dummy_data = {
            "messages": ["Greetings Sultan!\n\nAs your neighbor I would like to propose an alliance! What are your views on the board so far?", "I think an alliance would be great! Perhaps a dmz in the Black Sea would be a good idea to solidify this alliance?\n\nAs for my views on the board, my first moves will be Western into the Balkans and Mediterranean Sea.", "Sounds good lets call a dmz in the black sea", "Hey sorry for stabbing you earlier, it was an especially hard choice since Turkey is usually my country of choice. It's cool we got to do this study huh?"],
            "sender_labels": [False, True, False, True], # Example sender labels
            "receiver_labels": [True, True, False, "NOANNOTATION"], # Example receiver labels (True, False, NOANNOTATION)
            "speakers": ["russia", "turkey", "russia", "russia"],
            "receivers": ["turkey", "russia", "turkey", "turkey"],
            "absolute_message_index": [78, 107, 145, 717],
            "relative_message_index": [0, 1, 2, 3],
            "seasons": ["Spring", "Spring", "Spring", "Fall"],
            "years": ["1901", "1901", "1901", "1905"],
            "game_score": ["4", "3", "4", "7"],
            "game_score_delta": ["1", "-1", "1", "7"],
            "players": ["russia", "turkey"],
            "game_id": 10
        }
        with open(INPUT_FILE, 'w') as f:
            json.dump(dummy_data, f)
            f.write('\n') # Ensure it's a valid jsonlines file

    # Run the main processing function
    process_diplomacy_data(INPUT_FILE)
