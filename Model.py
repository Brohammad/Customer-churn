import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,balanced_accuracy_score,roc_auc_score,make_scorer, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import GridSearchCV

df=pd.read_excel("Telco_customer_churn.xlsx")
print(df['City'].unique())
df.drop(['Churn Label','Count','Churn Score','CLTV','Lat Long','Churn Reason','CustomerID', 'State','Country'],axis=1,inplace=True)
df['City'].replace(' ','_',regex=True,inplace=True)
df.columns = df.columns.str.replace(' ', '_')
print(df.head())
print(df['Total_Charges'].unique())

#df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])
print(len(df.loc[df['Total_Charges'] == ' ']))
df.loc[df['Total_Charges'] == ' ','Total_Charges'] = 0
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])
X=df.drop('Churn_Value',axis=1).copy()
print(X.head())
y=df['Churn_Value'].copy()
print(y.head())
print(X.dtypes  )
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
