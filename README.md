# ⚖️ Recidivism Risk Predictive Engine
### Supervised Machine Learning & Behavioral Risk Scoring Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classification-F7931E?style=flat)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP%20Feature%20Importance-blueviolet?style=flat)

---

## 🔍 Project Overview
Recidivism prediction is a cornerstone of forensic psychology and correctional risk assessment. Determining whether an individual has a high likelihood of re-offending requires balancing **historical behavioral indicators**, **prior offenses**, and **socio-environmental factors**.

This project implements a complete **end-to-end binary classification engine** to score recidivism risk. Beyond model performance, the project incorporates **Explainable AI (SHAP)** to inspect feature attribution and evaluate fairness across demographic groups.

---

## 🎯 Analytical Objectives
1. **Behavioral Risk Profiling:** Identify which features (priors count, age at offense, offense severity) have the highest predictive weight.
2. **Model Evaluation:** Optimize precision vs. recall tradeoff to minimize false positives while maintaining high detection.
3. **Model Explainability:** Deconstruct "black-box" predictions into interpretable risk factors using SHAP values.

---

## 🛠️ Tech Stack
- **Languages & Frameworks:** Python, Pandas, NumPy, Scikit-Learn
- **Models Evaluated:** Logistic Regression (Baseline), Random Forest Classifier
- **Model Interpretability:** SHAP (SHapley Additive exPlanations)
- **Evaluation Metrics:** ROC-AUC, Precision-Recall Curve, Confusion Matrix

---

## 💡 Key Behavioral Insights
- **The Escalation Threshold:** Prior juvenile offenses combined with rapid re-arrest cycles were the strongest behavioral indicators of near-term recidivism.
- **Explainability Over Accuracy:** In high-stakes human risk decisions, interpretable models are critical to prevent systemic bias and validate psychological validity.

---

## 🏢 Enterprise Application
- **Credit Risk & Default Modeling:** Evaluating loan default probability based on credit behavioral history.
- **Fraud Risk Scoring:** Real-time risk probability calculation for user accounts.
- **Employee & Customer Churn Modeling:** Early warning systems detecting disengagement and risk signals.

---
**Author:** [Prachi Kaushal](https://www.linkedin.com/in/prachi-kaushal-63a395231/) | Data Analyst & Behavioral Risk Modeler
