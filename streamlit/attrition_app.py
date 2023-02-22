# This code mimic from: https://www.youtube.com/watch?v=Eai1jaZrRDs&list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE&index=3

import streamlit as st
import pandas as pd
import numpy as np
import pickle
#from sklearn.ensemble import RandomForestClassifier

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

def salary_range(salary):
    if salary<2500:
        return 1250
    elif salary<5000:
        return 3750
    elif salary<7500:
        return 6250
    elif salary<10000:
        return 8750
    elif salary<12500:
        return 11250
    elif salary<15000:
        return 13750
    elif salary<17500:
        return 16250
    elif salary<20000:
        return 18750
    elif salary<25000:
        return 21250
    else:
        return 23750

st.write("""
#### Resignation Prediction (Use left panel for Input Features)
""")

st.sidebar.header('Input Features')

# Collects user input features into dataframe
uploaded_file = None
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age = st.sidebar.slider('Age', 18,59,37)
        business_travel = st.sidebar.radio('Business Travel', ('Yes', 'No'))
        business_travel = 1 if business_travel == 'Yes' else 0 #convert
        department = st.sidebar.selectbox('Department',('Research & Development', 'Human Resources', 'Sales'))
        distance = st.sidebar.slider('Distance from Home (miles)', 1,29,5)
        education_field = st.sidebar.selectbox('Education Field', ('Life Sciences', 'Human Resources', 'Marketing', 'Medical', 'Technical Degree', 'Other',))
        environment_satisfaction = st.sidebar.slider('Environment Satisfaction', 1,4,3)
        job_involvement = st.sidebar.slider('Job Involvement', 1,4,3)
        job_level = st.sidebar.slider('Job Level', 1,5,2)
        job_role = st.sidebar.selectbox('Job Role', ('Sales Executive','Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Representative'))#'JobRole', # category
        job_sat = st.sidebar.slider('Job Satisfaction', 1,4,3)#'JobSatisfaction',
        marital = st.sidebar.selectbox('Marital Status', ('Single', 'Married'))#'MaritalStatus',
        marital = 1 if marital == 'Married' else 0 #convert
        salary = st.sidebar.slider('Salary', 1000,20000,5000)
        num_comp_work_1 = st.sidebar.slider('Number of Company Worked', 1, 10, 2)#'NumCompaniesWorked',
        #'num_comp_work',
        ot = st.sidebar.radio('Overtime', ('Yes', 'No'))#'OverTime', # (*****+25%)
        relationship = st.sidebar.slider('RelationshipSatisfaction', 1,4,3)     
        stock = st.sidebar.slider('StockOptionLevel', 0, 3, 0) #'StockOptionLevel',
        ttl_wk_yr = st.sidebar.slider('TotalWorkingYears', 0,38,10) # (****-18%)
        training_times_last_year = st.sidebar.slider('TrainingTimesLastYear', 0,6,2)#'TrainingTimesLastYear',
        work_life_bal = st.sidebar.slider('WorkLifeBalance', 1, 4, 3)#'WorkLifeBalance', # (****4)
        year_at_comp = st.sidebar.slider('YearsAtCompany', 0, 37, 5)#'YearsAtCompany',
        year_in_cur_role = st.sidebar.slider('YearsInCurrentRole', 0, 18, 2)
        year_w_cur_mgr = st.sidebar.slider('YearsWithCurrManager', 0, 17, 2)
        
        data = {'Age': age,
                'BusinessTravel': business_travel,
                'Department': department,
                'DistanceFromHome': distance,
                'EducationField': education_field,
                'EnvironmentSatisfaction': environment_satisfaction,
                'JobInvolvement': job_involvement,
                'JobLevel': job_level,
                'JobRole': job_role,
                'JobSatisfaction': job_sat,
                'MaritalStatus': marital,
                'salary_range': salary_range(salary),
                'NumCompaniesWorked': num_comp_work_1,
                'num_comp_work': 2.5 if num_comp_work_1 < 5 else 7.5,
                'OverTime': 1 if ot == 'Yes' else 0,
                'RelationshipSatisfaction': relationship,
                'StockOptionLevel': stock,
                'TotalWorkingYears': ttl_wk_yr,
                'TrainingTimesLastYear': training_times_last_year,
                'WorkLifeBalance': work_life_bal,
                'YearsAtCompany': year_at_comp,
                'YearsInCurrentRole': year_in_cur_role,
                'YearsWithCurrManager': year_w_cur_mgr,
               }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

st.write(input_df.iloc[:,:8])
st.write(input_df.iloc[:,8:16])
st.write(input_df.iloc[:,16:25])
    
# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
#penguins_raw = pd.read_csv('penguins_cleaned.csv')
#penguins = penguins_raw.drop(columns=['species'])
#df = pd.concat([input_df,penguins],axis=0)


###############################################################
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

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

import os
#st.write(os.path.dirname(os.path.abspath('employee_data_clean_eda.csv')))

path_online = '/app/resignation/streamlit/employee_data_clean_eda.csv'
path_local = 'employee_data_clean_eda.csv'

if os.path.isfile(path_online):
    df = pd.read_csv(path_online)
    status = 'online'
else:
    df = pd.read_csv(path_local)
    status = 'local machine'

X = df[selected_features]
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.15)
X_test = pd.concat([input_df,X_test],axis=0)

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
score = m.predict_proba(X_test_ct)[0][1]

############## Result ##########################################
if score >= 0.5:
    new_title = f'<p style="font-family:sans-serif; color:Red; font-size: 24px;">Prediction: Tend to Leave</p>'
else:
    new_title = f'<p style="font-family:sans-serif; color:Green; font-size: 24px;">Prediction: Tend to Stay</p>'

st.markdown(new_title, unsafe_allow_html=True)
st.write(f'(score={round(score,3)})')
st.write(status)
#st.write('#### Prediction', '===>', '**Tend to "Leave"**' if score >= 0.5 else '**Tend to "Stay"**', f'(score={round(score,3)})')

###############################################################

#st.write(X_test[selected_features].iloc[:1,:])

