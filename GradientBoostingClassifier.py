#!/usr/bin/env python
# coding: utf-8

# ### Gradient Boosting Classifier

# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV , KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[22]:


# Load dataset
df = pd.read_csv('diabetes.csv')
df


# In[24]:


# Features and target
X = df.drop('class', axis = 1)
y = df['class']

# standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[26]:


# perform train, test split on the dataset
X_train, X_test, y_train , y_test = train_test_split(X_scaled,y, test_size = 0.8, random_state = 42)


# In[28]:


# Instanticate the model and define the parameters

gbc = GradientBoostingClassifier(random_state = 42)

# set up KFold cross _validation
kfold = StratifiedKFold(n_splits=5, shuffle= True, random_state=42)
# Hyperparameter grid
param_grid = {
    'n_estimators':[50,100,150],
    'learning_rate':[0.01,0.1,0.2],
    'max_depth':[3,4,5],
    'subsample':[0.8,1.0]
}
 # Grid serach with cross validation
grid_search = GridSearchCV(estimator=gbc,
                           param_grid=param_grid,
                           cv=kfold,
                           scoring='recall',
                           n_jobs=-1,
                           verbose=1)


# In[34]:


# fit the model
grid_search.fit(X_train, y_train)

#Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best Cross_validated Recall:", grid_search.best_score_)


# In[35]:


# Evaluate on test data uisng best estimator
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ## identify feature importance scores using XGBClassifier

# In[41]:


best_model.feature_importances_

