#!/bin/bash

# Script to run all implemented deception detection models

# Default parameters
MODEL="all"
TASK="sender"
POWER=false

# Function to display usage information
display_help() {
    echo "Usage: bash run_models.sh [options]"
    echo
    echo "Options:"
    echo "  -m, --model MODEL    Model to run: baselines, harbingers, bow, lstm, contextlstm, bertcontext, all (default: all)"
    echo "  -t, --task TASK      Task to perform: sender, receiver (default: sender)"
    echo "  -p, --power          Include power features (default: off)"
    echo "  -h, --help           Display this help message and exit"
    echo
    echo "Examples:"
    echo "  bash run_models.sh --model bow --task receiver --power"
    echo "  bash run_models.sh --model all --task sender"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -p|--power)
            POWER=true
            shift
            ;;
        -h|--help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

# Validate parameters
if [[ "$MODEL" != "baselines" && "$MODEL" != "harbingers" && "$MODEL" != "bow" && 
      "$MODEL" != "lstm" && "$MODEL" != "contextlstm" && "$MODEL" != "bertcontext" && 
      "$MODEL" != "all" ]]; then
    echo "Invalid model: $MODEL"
    display_help
    exit 1
fi

if [[ "$TASK" != "sender" && "$TASK" != "receiver" ]]; then
    echo "Invalid task: $TASK"
    display_help
    exit 1
fi

# Set power flag for Python commands
POWER_FLAG=""
if [ "$POWER" = true ]; then
    POWER_FLAG="--power"
fi

# Function to run model with current settings
run_model() {
    local model_name=$1
    local model_script=$2
    
    echo "=========================================="
    echo "Running $model_name model"
    echo "Task: $TASK"
    echo "Power features: $([ "$POWER" = true ] && echo "Yes" || echo "No")"
    echo "=========================================="
    
    python "$model_script" --task "$TASK" $POWER_FLAG
    
    echo -e "\n"
}

# Run selected model(s)
if [ "$MODEL" = "all" ] || [ "$MODEL" = "baselines" ]; then
    python implement_baselines.py
fi

if [ "$MODEL" = "all" ] || [ "$MODEL" = "harbingers" ]; then
    run_model "Harbingers" "implement_harbingers.py"
fi

if [ "$MODEL" = "all" ] || [ "$MODEL" = "bow" ]; then
    run_model "Bag of Words" "implement_bagofwords.py"
fi

if [ "$MODEL" = "all" ] || [ "$MODEL" = "lstm" ]; then
    run_model "LSTM" "implement_lstm.py"
fi

if [ "$MODEL" = "all" ] || [ "$MODEL" = "contextlstm" ]; then
    run_model "Context LSTM" "implement_contextlstm.py"
fi

if [ "$MODEL" = "all" ] || [ "$MODEL" = "bertcontext" ]; then
    run_model "BERT+Context" "implement_bertcontext.py"
fi

echo "All specified models completed!"
