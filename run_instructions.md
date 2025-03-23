# Running Deception Detection Models with a Virtual Environment

This guide provides step-by-step instructions for setting up a Python virtual environment and running the deception detection models.

## Setting Up a Virtual Environment

### For Windows

```bash
# Navigate to your project directory
cd d:\NLP\Deception_Detection

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

### For macOS/Linux

```bash
# Navigate to your project directory
cd /path/to/NLP/Deception_Detection

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

## Running the Models

After activating your virtual environment and installing all dependencies, you can run each model using the following commands:

### 1. Baseline Models
The baselines model doesn't require task or power parameters:

```bash
python implement_baselines.py
```

### 2. Harbingers Model

Run on sender task (actual lie detection):
```bash
python implement_harbingers.py --task sender
```

Run with power features:
```bash
python implement_harbingers.py --task sender --power
```

Run on receiver task (suspected lie detection):
```bash
python implement_harbingers.py --task receiver
```

Run on receiver task with power features:
```bash
python implement_harbingers.py --task receiver --power
```

### 3. Bag of Words Model

Run on sender task:
```bash
python implement_bagofwords.py --task sender
```

Run with power features:
```bash
python implement_bagofwords.py --task sender --power
```

Run on receiver task:
```bash
python implement_bagofwords.py --task receiver
```

Run on receiver task with power features:
```bash
python implement_bagofwords.py --task receiver --power
```

### 4. LSTM Model

Run on sender task:
```bash
python implement_lstm.py --task sender
```

Run with power features:
```bash
python implement_lstm.py --task sender --power
```

Run on receiver task:
```bash
python implement_lstm.py --task receiver
```

Run on receiver task with power features:
```bash
python implement_lstm.py --task receiver --power
```

### 5. ContextLSTM Model

Run on sender task:
```bash
python implement_contextlstm.py --task sender
```

Run with power features:
```bash
python implement_contextlstm.py --task sender --power
```

Run on receiver task:
```bash
python implement_contextlstm.py --task receiver
```

Run on receiver task with power features:
```bash
python implement_contextlstm.py --task receiver --power
```

### 6. BERT+Context Model

Run on sender task:
```bash
python implement_bertcontext.py --task sender
```

Run with power features:
```bash
python implement_bertcontext.py --task sender --power
```

Run on receiver task:
```bash
python implement_bertcontext.py --task receiver
```

Run on receiver task with power features:
```bash
python implement_bertcontext.py --task receiver --power
```

## Running All Models with a Batch Script

Windows users can utilize a batch script for running all models:

### d:\NLP\Deception_Detection\run_models.bat

Create a file `run_models.bat` with the following content:

```batch
@echo off
echo Running Deception Detection Models...

set TASK=sender
set POWER=

if "%1"=="receiver" set TASK=receiver
if "%2"=="power" set POWER=--power

echo.
echo Running Baselines...
python implement_baselines.py

echo.
echo Running Harbingers Model with task=%TASK% power=%POWER%...
python implement_harbingers.py --task %TASK% %POWER%

echo.
echo Running Bag of Words Model with task=%TASK% power=%POWER%...
python implement_bagofwords.py --task %TASK% %POWER%

echo.
echo Running LSTM Model with task=%TASK% power=%POWER%...
python implement_lstm.py --task %TASK% %POWER%

echo.
echo Running ContextLSTM Model with task=%TASK% power=%POWER%...
python implement_contextlstm.py --task %TASK% %POWER%

echo.
echo Running BERT+Context Model with task=%TASK% power=%POWER%...
python implement_bertcontext.py --task %TASK% %POWER%

echo.
echo All models completed!
```

Example usage:
```bash
# Run all models with default settings (sender task, no power)
run_models.bat

# Run all models for receiver task
run_models.bat receiver

# Run all models for receiver task with power features
run_models.bat receiver power
```

## Troubleshooting

1. **Memory Issues with BERT Models**:
   - If you encounter memory errors when running BERT models, try reducing the batch size in the code (e.g., change `batch_size = 2` to `batch_size = 1`)
   - You can also try running on a machine with more RAM or GPU memory

2. **GloVe Embedding Download Issues**:
   - If the automatic download of GloVe embeddings fails, manually download them from https://nlp.stanford.edu/projects/glove/
   - Place the downloaded file in the `d:\NLP\Deception_Detection\embeddings_cache` directory

3. **CUDA Issues**:
   - If you have a CUDA-compatible GPU but PyTorch doesn't detect it, verify that:
     - You have the correct CUDA toolkit installed
     - You installed the CUDA-enabled version of PyTorch
     - Your GPU drivers are up to date

4. **Package Installation Issues**:
   - If you experience issues installing packages, try installing them individually:
   ```bash
   pip install torch
   pip install transformers
   pip install scikit-learn
   ```
