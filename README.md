# Euphemism Detection with Fine-Tuned XLNet

A text classification project that detects **euphemisms** in English text using a fine-tuned XLNet transformer model — distinguishing sentences that use indirect, softened language from those that use direct, literal language.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model](#model)
- [Training Setup](#training-setup)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [Related Publication](#related-publication)
- [License](#license)

---

## 🩺 Overview

Euphemisms are a subtle but common form of figurative language — phrases used to soften or disguise the directness of a statement (e.g. using "passed away" instead of "died"). Automatically detecting euphemistic language is useful for content moderation, sentiment analysis, and understanding nuanced communication, but it's a genuinely hard NLP problem because euphemisms rely heavily on context rather than fixed keywords.

This project fine-tunes a pretrained **XLNet** (`xlnet-base-cased`) transformer model to perform binary classification: given a sentence, predict whether it contains a **euphemism** or **not**. The pipeline covers the full workflow — from raw text preprocessing through training, evaluation, and generating predictions on unseen data.

---

## 📊 Dataset

The project uses labeled English-language text data for euphemism classification:

| File | Purpose |
|---|---|
| `EN_train.csv` | Training data with `text` and `label` columns |
| `EN_test.csv` | Unlabeled test data used to generate predictions |
| `EN_reference_test.csv` | Ground-truth labels for the test set, used for final evaluation |

Labels are binary: **Euphemism** vs. **Non-Euphemism**.

---

## 🔄 Pipeline

### 1. Data Preprocessing
- Drop rows with missing values
- Lowercase all text
- Strip special characters, keeping only alphanumeric characters and whitespace
- Encode string labels into numeric form with `LabelEncoder`

### 2. Tokenization
- Tokenize text using the `XLNetTokenizer` (`xlnet-base-cased`)
- Pad/truncate all sequences to a maximum length of 512 tokens
- Split data into training (80%) and validation (20%) sets

### 3. Model Fine-Tuning
- Load `XLNetForSequenceClassification` with 2 output labels
- Fine-tune using Hugging Face's `Trainer` API, with experiment tracking via **Weights & Biases (wandb)**

### 4. Evaluation & Testing
- Evaluate on the held-out validation split during training
- Generate predictions on the separate test set
- Compare predictions against reference labels to compute final test accuracy and a full classification report

### 5. Inference on New Data
- A reusable `predict_labels()` function loads the saved fine-tuned model and tokenizer, and labels any new CSV file of text samples on demand

---

## 🤖 Model

**Base model:** `xlnet-base-cased` (Hugging Face Transformers)
**Task:** Binary sequence classification (Euphemism / Non-Euphemism)
**Framework:** PyTorch + Hugging Face `Trainer`

XLNet was chosen over more common models like BERT because its autoregressive, permutation-based pretraining captures bidirectional context without the independence assumptions of masked language models — potentially advantageous for the subtle contextual cues euphemisms depend on.

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Train batch size | 16 |
| Eval batch size | 16 |
| Learning rate | 2e-5 |
| Warmup steps | 1,000 |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Metric for best model | F1 score (maximized) |
| Evaluation/save strategy | Per epoch, best model reloaded at end |
| Experiment tracking | Weights & Biases (wandb) |

---

## 📈 Evaluation

Model performance is assessed with:
- **Accuracy, precision, recall, F1-score** (via `sklearn.metrics`)
- **Confusion matrix**, visualized as a Seaborn heatmap for both the validation split and the held-out test set
- **Classification report** breaking down performance per class (Euphemism vs. Non-Euphemism)

The trained model and tokenizer are saved locally so they can be reloaded for inference without retraining.

---

## 🧰 Tech Stack

- **Language:** Python 3
- **NLP / Deep Learning:** Hugging Face Transformers, PyTorch
- **Data Handling:** Pandas, Hugging Face `datasets`
- **ML Utilities:** Scikit-learn (label encoding, metrics)
- **Visualization:** Matplotlib, Seaborn
- **Experiment Tracking:** Weights & Biases (wandb)
- **Environment:** Google Colab

---

## 📁 Project Structure

```
euphemism-detection/
│
├── euphemism_detection.ipynb       # Main notebook: preprocessing, training, evaluation, inference
├── data/
│   ├── EN_train.csv                # Training data
│   ├── EN_test.csv                 # Test data (unlabeled)
│   └── EN_reference_test.csv       # Ground-truth labels for test set
├── results/                        # Model checkpoints saved during training
├── XLNet_model/                    # Final fine-tuned model + tokenizer
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install transformers torch datasets scikit-learn pandas seaborn matplotlib wandb
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/euphemism-detection.git
   cd euphemism-detection
   ```
2. Place `EN_train.csv`, `EN_test.csv`, and `EN_reference_test.csv` in the `data/` folder, updating the file paths in the notebook to match your local setup (the original paths point to Google Drive).
3. Set up a free [Weights & Biases](https://wandb.ai) account if you want experiment tracking, or remove `report_to=["wandb"]` from the training arguments to skip it.
4. Open and run the notebook:
   ```bash
   jupyter notebook euphemism_detection.ipynb
   ```
5. To label new, unlabeled text data with the trained model, run the inference cell and provide the path to your CSV file when prompted.

> **Note:** Fine-tuning a transformer model benefits significantly from GPU acceleration. A Colab GPU runtime (or equivalent) is recommended.

---

## 💡 Key Insights

- Euphemism detection is fundamentally a **context-dependent** task — the same surface-level words can be euphemistic or literal depending on usage, which makes it a good fit for transformer-based contextual embeddings rather than keyword or rule-based approaches.
- Tracking the **F1 score** (rather than accuracy alone) as the model-selection metric matters here, since euphemism datasets are often imbalanced and F1 better reflects performance on the minority class.
- Separating the **validation** (used during training for model selection) from a fully **held-out test set** with independent reference labels gives a more honest measure of generalization than validation performance alone.

---

## 🔮 Future Improvements

- Experiment with newer transformer backbones (e.g. **DeBERTa**, **XLM-RoBERTa**) and compare against XLNet in an ensemble, as explored in the related publication below.
- Add **cross-validation** to reduce variance in the reported metrics.
- Perform **error analysis** on misclassified examples to identify systematic failure patterns (e.g. specific euphemism categories the model struggles with).
- Package the trained model with a simple **Gradio** interface for interactive testing, similar to other projects in this portfolio.
- Clean up the notebook by removing commented-out/duplicate training argument blocks before sharing publicly.

---

## 📄 Related Publication

This project's approach connects to ensemble-based euphemism detection research:

**Blend-XED: A Transformer-Based Blending Ensemble Model for Euphemism Detection**
Oindrela, D., Faiza, S. R., Islam, T., Chy, A. N., & Ahmed, T. — Accepted and published at the IEEE International Black Sea Conference on Communications and Networking (BlackSeaCom), Bucharest, Romania, June 2026.

---

## 📄 License

This project is open-source and available for educational and research purposes. Please cite appropriately if reused.
