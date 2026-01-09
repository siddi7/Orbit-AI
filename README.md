# ORBIT-Data Analyst

## Summary
Automated EDA, robust modeling, and explainability for small-sample socio-economic datasets. Built with Streamlit, scikit-learn, and Plotly.

## Quick run (local)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Upload your CSV and click "Run AutoAnalyst"

## Features
- **Auto-ML**: Trains Linear, Ridge, Lasso, RandomForest, and XGBoost (if available).
- **Robust CV**: Uses LeaveAudio-One-Out (LOO) for small datasets (n<=40).
- **Explainability**: SHAP plots and Feature Importance.
- **Reporting**: Auto-generates `executive_report.md`.
