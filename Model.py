import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,balanced_accuracy_score,roc_auc_score,make_scorer, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import GridSearchCV

df=pd.read_csv("Teleco_Customer_Churn.csv")
print(df.head())