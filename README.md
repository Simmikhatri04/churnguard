# 📉 ChurnGuard: Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave, explains why, and suggests retention actions. Built with Python, scikit-learn and Streamlit.

**🔗 Live app:** https://churnguard-rggrzf5zzvxz4b2j3ga9ru.streamlit.app/
**👤 Author:** Simran Khatri

## Problem
Acquiring a new customer costs far more than keeping an existing one. ChurnGuard helps a telecom company find at-risk customers early so it can act before they leave.

## Dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data, Kaggle): 7,043 customers, 21 columns. After cleaning, 7,032 rows remain. The churn rate is 26.6%, so the classes are imbalanced.

## Approach
1. **Cleaning:** converted `TotalCharges` to numeric, dropped 11 incomplete rows and the ID column, and one-hot encoded categories.
2. **EDA:** churn is highest for month-to-month contracts (about 43%) and for new customers (median churner tenure is about 10 months).
3. **Modeling:** compared Logistic Regression, Random Forest and XGBoost with class balancing, then tuned Random Forest with randomized search and 5-fold cross-validation.
4. **Threshold:** chose a decision threshold of **0.45** to favor recall, because missing a churner costs more than a wasted retention offer.
5. **Explainability:** SHAP values show which features drive each prediction.

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.491 | 0.797 | 0.608 | 0.835 |
| Random Forest | 0.546 | 0.746 | 0.631 | 0.836 |
| XGBoost | 0.493 | 0.783 | 0.605 | 0.834 |
| **Random Forest (tuned)** | 0.487 | 0.840 | 0.616 | **0.841** |

The last row uses the 0.45 threshold. The others use the default 0.5. Accuracy alone is misleading here, since always predicting "no churn" would already score about 73%.

**Top drivers (SHAP):** tenure, fiber-optic internet service and a two-year contract.

## Dashboard features
- **Overview:** churn rate and key charts
- **Predict:** churn risk for a single customer with a suggested action
- **Batch Scoring:** upload a CSV, get risk levels, and download results
- **Model Performance:** model comparison and threshold rationale
- **Key Drivers:** feature importance and SHAP summary

## Tech stack
Python, pandas, scikit-learn, XGBoost, SHAP, Plotly, Streamlit, Google Colab, GitHub.

## Run locally
```bash
git clone https://github.com/Mohitchoudhary789/churnguard.git
cd churnguard
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
app.py                 Streamlit dashboard
requirements.txt       Dependencies
churn_model.pkl        Trained Random Forest
columns.pkl            Feature columns used by the model
config.json            Decision threshold
clean_churn.csv        Cleaned dataset
model_comparison.csv   Model results table
shap_summary.png       SHAP summary plot
*.ipynb                Training notebook
```

## Limitations and future work
- The dataset is a single snapshot, so the model can't capture changes over time.
- The threshold was chosen on the test split; a separate validation set would be more rigorous.
- Future work: MLflow experiment tracking, drift monitoring, a Docker image and CI/CD.
