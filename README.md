# Deception Detection in Diplomacy Game

This repository contains implementations of different models for detecting deception in the Diplomacy game dataset as described in the ACL 2020 paper ["It Takes Two to Lie: One to Lie, and One to Listen"](https://www.aclweb.org/anthology/2020.acl-main.353/).

## Overview

This project explores computational models for detecting deception in human-human text conversations from the game of Diplomacy. We implement various models ranging from simple baselines to advanced deep learning approaches, comparing their performance on detecting both actual lies (sender's intention) and suspected lies (receiver's perception).

## Installation

### Requirements

1. Python 3.7+
2. PyTorch 1.7+
3. Transformers 4.5+
4. scikit-learn
5. spaCy
6. gensim

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deception_detection.git
cd deception_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Data

The project uses the Diplomacy dataset from the original paper. The data should be placed in the following structure:
```
d:\NLP\Deception_Detection\
└── 2020_acl_diplomacy-master\
    └── data\
        ├── train.jsonl
        ├── validation.jsonl
        └── test.jsonl
```

## Models

This project implements the following models for deception detection:

1. **Baselines**: Random and majority-class baselines
2. **Harbingers**: Linguistic feature-based model using lexical markers
3. **Bag of Words**: Classical BoW with logistic regression
4. **LSTM**: Basic LSTM model with GloVe embeddings
5. **ContextLSTM**: Hierarchical LSTM model that captures conversation context
6. **BERT+Context**: BERT embeddings with a context-aware LSTM layer

## Running the Models

You can run individual models or all models using the provided scripts.

### Using the Bash Script (for Unix/Linux/Mac)

```bash
# Run all models with default settings (sender task, no power features)
bash run_models.sh --run-all

# Run specific model (e.g., LSTM) on sender task with power features
bash run_models.sh --model lstm --task sender --power

# Run specific model on receiver task
bash run_models.sh --model bertcontext --task receiver 

# Display help
bash run_models.sh --help
```

### Running Individual Models

#### Baseline Models

```python
# Run random and majority baselines
python implement_baselines.py
```

#### Harbingers Model 

```python
# Run on sender task (actual lie detection)
python implement_harbingers.py --task sender

# Run with power features
python implement_harbingers.py --task sender --power

# Run on receiver task (suspected lie detection)
python implement_harbingers.py --task receiver

# Run on receiver task with power features
python implement_harbingers.py --task receiver --power
```

#### Bag of Words Model

```python
# Run on sender task
python implement_bagofwords.py --task sender

# Run with power features
python implement_bagofwords.py --task sender --power

# Run on receiver task
python implement_bagofwords.py --task receiver

# Run on receiver task with power features
python implement_bagofwords.py --task receiver --power
```

#### LSTM Model

```python
# Run on sender task
python implement_lstm.py --task sender

# Run with power features
python implement_lstm.py --task sender --power

# Run on receiver task
python implement_lstm.py --task receiver

# Run on receiver task with power features
python implement_lstm.py --task receiver --power
```

#### ContextLSTM Model

```python
# Run on sender task
python implement_contextlstm.py --task sender

# Run with power features
python implement_contextlstm.py --task sender --power

# Run on receiver task
python implement_contextlstm.py --task receiver

# Run on receiver task with power features
python implement_contextlstm.py --task receiver --power
```

#### BERT+Context Model

```python
# Run on sender task
python implement_bertcontext.py --task sender

# Run with power features
python implement_bertcontext.py --task sender --power

# Run on receiver task
python implement_bertcontext.py --task receiver

# Run on receiver task with power features
python implement_bertcontext.py --task receiver --power
```

## Model Performance

Each model's performance is evaluated using:
- Accuracy
- Macro F1 score
- Binary/Lie F1 score (F1 for the deceptive class)
- Precision & Recall

Results are automatically compared to the original paper's reported metrics.

## Tasks

The models can be evaluated on two different tasks:

1. **Sender Task** (actual lie detection): Predict whether the sender is being truthful or deceptive based on their actual intention
2. **Receiver Task** (suspected lie detection): Predict whether the receiver perceives a message as truthful or deceptive

## Features

All models have the option to incorporate **power features**, which capture the game state:
- Score delta between sender and receiver
- Binary features for severe power imbalances (>4 supply centers)

## References

Peskov, D., Cheng, B., Elgohary, A., Barrow, J., Danescu-Niculescu-Mizil, C., & Boyd-Graber, J. (2020). It Takes Two to Lie: One to Lie, and One to Listen. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 3811-3854).

```
@inproceedings{peskov-etal-2020-takes,
    title = "It Takes Two to Lie: One to Lie, and One to Listen",
    author = "Peskov, Denis and Cheng, Benny and Elgohary, Ahmed and Barrow, Joe and Danescu-Niculescu-Mizil, Cristian and Boyd-Graber, Jordan",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    pages = "3811--3854"
}