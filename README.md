# ICL4RUL

Remaining Useful Life (RUL) estimation using In-Context Learning and Deep Learning architectures (BiLSTM) evaluated on the NASA C-MAPSS turbofan engine degradation dataset.

## Project Structure

```
ICL4RUL/
├── data/              # C-MAPSS dataset (FD001, etc.)
├── models/            # Saved model checkpoints
├── notebooks/         # Exploratory notebooks
├── results/           # Evaluation metrics and outputs
├── src/               # Preprocessing and model training code
│   ├── preprocessing.py
│   └── baseline_bilstm.py
├── .gitignore
└── README.md
```

## Setup & Usage

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

2. Run preprocessing and baseline:
   ```bash
   python src/baseline_bilstm.py
   ```
