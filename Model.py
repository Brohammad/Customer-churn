import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, roc_auc_score, make_scorer, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("Telco_customer_churn.xlsx")
print(df['City'].unique())

# Drop unnecessary columns
df.drop(['Churn Label','Count','Churn Score','CLTV','Lat Long','Churn Reason','CustomerID', 'State','Country'], axis=1, inplace=True)

# Clean column names and values
df['City'].replace(' ','_', regex=True, inplace=True)
df.columns = df.columns.str.replace(' ', '_')
print(df.head())

# Handle Total_Charges
print(df['Total_Charges'].unique())
print(len(df.loc[df['Total_Charges'] == ' ']))
df.loc[df['Total_Charges'] == ' ', 'Total_Charges'] = 0
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])

# Split features and labels
X = df.drop('Churn_Value', axis=1).copy()
y = df['Churn_Value'].copy()
print(X.head())
print(y.head())
print(X.dtypes)

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=[
    'City',
    'Gender',
    'Senior_Citizen',
    'Partner',
    'Dependents',
    'Phone_Service',
    'Multiple_Lines',
    'Internet_Service',
    'Online_Security',
    'Online_Backup',
    'Device_Protection',
    'Tech_Support',
    'Streaming_TV',
    'Streaming_Movies',
    'Contract',
    'Paperless_Billing',
    'Payment_Method'
])

print(X_encoded.head())
print(sum(y)/len(y))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(sum(y_train)/len(y_train))    
print(sum(y_test)/len(y_test))

# Booster (low-level API)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'eta': 0.1,
    'max_depth': 6
}

evallist = [(dtrain, 'train'), (dvalid, 'eval')]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evallist,
    early_stopping_rounds=10,
    verbose_eval=True
)

# Predict using bst (booster)
y_pred = bst.predict(dvalid)
y_pred = (y_pred > 0.5).astype(int)

# Confusion matrix for bst
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    values_format='d',
    display_labels=["Did not leave", "Left"]
)
plt.show()

# Grid Search param grid
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.2, 0.3],
    'gamma': [0.1, 0.2, 0.3],
    'reg_lambda': [0, 1, 10.0],
    'scale_pos_weight': [1, 3, 5]
}

# Final XGBClassifier after tuning (no early stopping)
clf_xgb = xgb.XGBClassifier(
    seed=42,
    objective='binary:logistic',
    gamma=0.25,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=10,
    scale_pos_weight=3,
    subsample=0.9,
    colsample_bytree=0.5,
    eval_metric='aucpr'  # Moved here from .fit()
)


clf_xgb.fit(
    X_train,
    y_train,
    verbose=True,
    eval_set=[(X_test, y_test)]
    # Removed eval_metric from here
)

# Final Confusion Matrix (from classifier)
ConfusionMatrixDisplay.from_estimator(
    clf_xgb,
    X_test,
    y_test,
    values_format='d',
    display_labels=["Did not leave", "Left"],
    cmap='Blues'
)
plt.show()
