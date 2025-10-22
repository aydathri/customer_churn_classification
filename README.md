# Customer Churn Prediction (ML Project)

This project predicts customer churn using multiple machine learning models and compares their performance.  
The dataset includes customer information such as demographics, service details, and contract types.  
The goal is to identify which customers are likely to leave the company.

---

## Project Workflow

1. **Data Preprocessing**
   - Handling missing values (using `IterativeImputer`)
   - Encoding categorical features (`OrdinalEncoder`, `OneHotEncoder`)
   - Scaling numerical features (`MinMaxScaler`, `StandardScaler`)
   - Applying PCA (for certain models)
   - Balancing data with `SMOTE`

2. **Model Training**
   Models used:
   - Logistic Regression  
   - Support Vector Classifier (SVC)  
   - K-Nearest Neighbors (KNN)  
   - Random Forest (RF)  
   - XGBoost (XGB)  
   - Voting Classifier  

3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC  
   - Confusion Matrix  
   - ROC Curves  
   - Model Comparison Table and Bar Chart  

4. **Overfitting Analysis**
   - Learning Curves for all models  
   - Validation Curves (for non-ensemble models)  

---

## Dataset
- Format: CSV
- Path: `https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn`

---

## Project Structure

CustomerChurnML/
│
├── telco-customer-churn.ipynb    # Main Jupyter Notebook
├── pipeline.html                 # Pipeline visualization
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies

---

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
---
## Key Libraries
scikit-learn for ML models, preprocessing, and evaluation

imbalanced-learn for handling imbalanced data

xgboost for gradient boosting

matplotlib, seaborn, plotly for visualization

pandas, numpy for data manipulation

---

## Results
Models are compared based on ROC-AUC and other metrics.

Visualizations help to identify overfitting and model performance clearly.

---
### Author
#### Ayda Taheri✨