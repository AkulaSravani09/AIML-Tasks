#!/usr/bin/env python
# coding: utf-8

# - It was developed in 2016 by Tianqi Chen and Carlos Guestrin.
# - XGB Boost and Gradient Boosting Machines are both ensemble tree methods.
# - XGB boosting is the latest.
# - Spars data ccan be handled by using XG boost.

# In[2]:


# pip install xgboost


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# In[4]:


# load data
df = pd.read_csv('diabetes.csv')
df


# In[5]:


# split features and target
X = df.drop('class', axis=1)
y = df['class']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[6]:


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)
print("-----------------------------------------------------------------")
print(X_test_scaled)


# In[19]:


# XGBoost classifier Instantiation with hyper parameter grid

xgb = XGBClassifier(use_label_encoder=False, eval_metrics='logloss', random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 150, 200, 300],
    'learning_rate': [0.01, 0.1, 0.15],
    'max_depth': [2,3,4,5],
    'subsample': [0.8,1.0],
    'colsample_bytree':[0.8, 1.0]
}

 # Stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCv with scoring = recall
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='recall',
                           cv=skf,
                           verbose=1,
                           n_jobs=-1)


# In[26]:


# fit the model
grid_search.fit(X_train, y_train)

# Find the best model, best cross validated recall score
#Best parameters and score

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best Cross_validated Recall:", grid_search.best_score_)
# predictions on test set
y_pred = best_model.predict(X_test)


# In[30]:


# evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




