import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
pd.options.display.max_columns = None

# select one that correlation over +/- 10%
selected_features = [
    'Age',
    'BusinessTravel',
    'Department',
    'DistanceFromHome',
    'EducationField',
    'EnvironmentSatisfaction',
    'JobInvolvement',
    'JobLevel',
    'JobRole', # category
    'JobSatisfaction',
    'MaritalStatus',
    'salary_range', # will test between MonthlyIncome
    'NumCompaniesWorked',
    'num_comp_work',
    'OverTime', # (*****+25%)
    'RelationshipSatisfaction',
    'StockOptionLevel',
    'TotalWorkingYears', # (****-18%)
    'TrainingTimesLastYear',
    'WorkLifeBalance', # (****4)
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsWithCurrManager'
]

df = pd.read_csv('../data/employee_data_clean_eda.csv')

X = df[selected_features]
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.15)
glimpse = []
for c in df.columns:
    make_set = list(set(df[c]))
    data = {'column_name':c, 'set_10':make_set[:10], 'set_count':len(make_set), 'type':str(df[c].dtypes)}
    #glimpse.append(data)
    if data['type'] == 'object':
        glimpse.append(data)
df_glimpse = pd.DataFrame(glimpse)
dummy_col = [c for c in df_glimpse['column_name'] if c in selected_features]

numerical_features = [c for c in selected_features if c not in dummy_col]

ct = ColumnTransformer([
        ('sc', StandardScaler(), numerical_features),
        ('ohe', OneHotEncoder(), dummy_col)
    ], remainder='passthrough')

X_train_ct = ct.fit_transform(X_train[selected_features])
X_test_ct = ct.transform(X_test[selected_features])

m = AdaBoostClassifier(random_state=42)
m.fit(X_train_ct, y_train)
print(m.score(X_train_ct, y_train))
print(m.score(X_test_ct, y_test))
print(confusion_matrix(y_train, m.predict(X_train_ct)))
print(confusion_matrix(y_test, m.predict(X_test_ct)))