# Fraud_detection

# Fraud Transaction Detection (6.36M rows)

A production‑ready ML pipeline to detect fraudulent financial transactions using Python and scikit‑learn.

## Key Results
- ROC‑AUC: 0.992
- Fraud recall: 0.88
- Fraud precision: 0.93

## Highlights
- Leak‑aware feature engineering: merchant flag, transaction type (one‑hot), balance mismatch/error signals
- Severe class imbalance handled via SMOTE + class_weight
- Tuned RandomForest for high recall with controlled false positives
- Clear evaluation: ROC/PR curves, confusion matrix, classification report
- Explainability: feature importances and SHAP insights (TRANSFER/CASH_OUT, high amounts, zeroed balances)

## Project Structure
- `fraud_detection.ipynb` — full E2E pipeline
- `Fraud.csv` — dataset (sample or link)
- `Data Dictionary.txt` — field descriptions

## Quick Start
1. Install: `pip install -r requirements.txt`
2. Run notebook: open `fraud_detection.ipynb`
3. Review outputs: metrics, curves, and SHAP explanations

## Tech
Python, pandas, scikit‑learn, imbalanced‑learn (SMOTE), matplotlib/seaborn, SHAP
