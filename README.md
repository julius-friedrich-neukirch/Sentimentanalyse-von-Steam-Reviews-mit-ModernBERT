# Steam Review Sentiment Analysis with ModernBERT

Fine-tuning of transformer-based models for binary sentiment 
classification of Steam game reviews (positive / negative).  
Bachelor's thesis project — Business Analytics, Friedrich Schiller 
University Jena, 2025.

---

## Overview

Steam users label their own reviews as "Recommended" or 
"Not Recommended". This project leverages these labels as 
ground truth to train and evaluate several transformer models 
on a large-scale, real-world NLP classification task.

The main focus is on **ModernBERT**, a recently released 
encoder model, benchmarked against established baselines 
such as RoBERTa, DeBERTa, and BERT.

---

## Results

Multiple models and hyperparameter configurations were evaluated.  
Best results per model (Accuracy / F1):

| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| ModernBERT-large    | 94.21%   | 94.15%   |
| DeBERTa-v3-base     | 93.91%   | 93.86%   |
| **ModernBERT-base** | **93.79%** | **93.76%** |
| RoBERTa-base        | 93.57%   | 93.52%   |
| BERT-base-uncased   | 93.13%   | 93.07%   |
| distilroberta-base  | 92.99%   | 92.93%   |

Best ModernBERT-base configuration:
- Learning rate: 5e-05
- Batch size: 64
- Epochs: 2
- Max length: 128

---

## Dataset

- Source: Steam game reviews via the Steam API
- Size: ~1,870,000 reviews
- Labels: Recommended / Not Recommended (binary)
- Split: 90% training / 10% validation

---

## Model & Approach

- Framework: HuggingFace Transformers + PyTorch
- Optimizer: AdamW
- Task: Binary sequence classification
- Multiple models benchmarked across varying learning rates,
  batch sizes, and sequence lengths

---

## Files

| File | Description |
|------|-------------|
| `Sentiment_Analysis.py` | Main training and evaluation script |
| `Sentiment_Analysis_Notebook.ipynb` | Notebook version with inline output |
| `Batch_Ausgabe.ipynb` | Visualization of batch training results |
| `data_log_alle_Ergebnisse.txt` | Raw experiment logs (all runs) |

---

## Requirements

```bash
pip install transformers torch pandas scikit-learn
```

---

## Author

Julius Neukirch — [LinkedIn](https://www.linkedin.com/in/julius-friedrich-neukirch) · [GitHub](https://github.com/julius-friedrich-neukirch)
