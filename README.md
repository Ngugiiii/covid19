import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set(rc={'figure.figsize':(14,8)}, font_scale=.9)
df = pd.read_csv('/kaggle/input/covid19-symptoms-checker/Cleaned-Data.csv')
display(df)
Fever	Tiredness	Dry-Cough	Difficulty-in-Breathing	Sore-Throat	None_Sympton	Pains	Nasal-Congestion	Runny-Nose	Diarrhea	...	Gender_Male	Gender_Transgender	Severity_Mild	Severity_Moderate	Severity_None	Severity_Severe	Contact_Dont-Know	Contact_No	Contact_Yes	Country
0	1	1	1	1	1	0	1	1	1	1	...	1	0	1	0	0	0	0	0	1	China
1	1	1	1	1	1	0	1	1	1	1	...	1	0	1	0	0	0	0	1	0	China
2	1	1	1	1	1	0	1	1	1	1	...	1	0	1	0	0	0	1	0	0	China
3	1	1	1	1	1	0	1	1	1	1	...	1	0	0	1	0	0	0	0	1	China
4	1	1	1	1	1	0	1	1	1	1	...	1	0	0	1	0	0	0	1	0	China
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
316795	0	0	0	0	0	1	0	0	0	0	...	0	1	0	0	0	1	0	1	0	Other
316796	0	0	0	0	0	1	0	0	0	0	...	0	1	0	0	0	1	1	0	0	Other
316797	0	0	0	0	0	1	0	0	0	0	...	0	1	0	0	1	0	0	0	1	Other
316798	0	0	0	0	0	1	0	0	0	0	...	0	1	0	0	1	0	0	1	0	Other
316799	0	0	0	0	0	1	0	0	0	0	...	0	1	0	0	1	0	1	0	0	Other
316800 rows × 27 columns

indicators = ['Fever', 'Tiredness', 'Dry-Cough',  'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion',
              'Runny-Nose', 'Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+', 'Gender_Male',
              'Gender_Female', 'Gender_Transgender']
target_columns = ['Severity_None']
indicators2 = ['Fever', 'Tiredness', 'Dry-Cough',  'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion',
              'Runny-Nose', 'Diarrhea', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+', 'Gender_Male',
              'Gender_Female', 'Gender_Transgender', 'Severity_None']
features = df[indicators]
targets = df[target_columns]
display(features.head(), targets.head())
Fever	Tiredness	Dry-Cough	Difficulty-in-Breathing	Sore-Throat	Pains	Nasal-Congestion	Runny-Nose	Diarrhea	Age_0-9	Age_10-19	Age_20-24	Age_25-59	Age_60+	Gender_Male	Gender_Female	Gender_Transgender
0	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0
1	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0
2	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0
3	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0
4	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0
Severity_None
0	0
1	0
2	0
3	0
4	0
# condition = []
# cond_dict = {
#     0: "Mild",
#     1: "Moderate",
#     2: "Severe"
# }
# for i in targets.values:
#     idx = np.where(i == 1)[0][0]
#     condition.append(cond_dict[idx])
# targets['Condition'] = condition
sns.set(rc={'figure.figsize':(12,8)}, font_scale=.9)
targets = targets.rename(columns={'Severity_None':'Non_Severe'})
sns.countplot(targets['Non_Severe'])
plt.title("Severity Data Distribution")
plt.show()
sns.set(rc={'figure.figsize':(12,8)}, font_scale=.9)
/opt/conda/lib/python3.7/site-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  FutureWarning

temp = []
for i in indicators:
    temp.append(sum(features[i].values))
temp_df = pd.DataFrame({"Indicator":indicators, "Occurence_Count":temp})
sns.barplot(data = temp_df, y="Indicator", x="Occurence_Count")
<AxesSubplot:xlabel='Occurence_Count', ylabel='Indicator'>

plt.pie(data=temp_df, x="Occurence_Count", labels=temp_df["Indicator"])
plt.show()

def get_symptom_count(the_list):
    return sum(the_list.values)
features['Total_Symptom'] = features[indicators].apply(get_symptom_count, axis=1)
feats = df[indicators2]
feats['Total_Symptom'] = feats[indicators].apply(get_symptom_count, axis=1)
/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  This is separate from the ipykernel package so we can avoid doing imports until
/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  """
sns.countplot(data=feats, x='Total_Symptom', hue='Severity_None')
plt.xlabel("Total symptom occurence on someone")
plt.show()

data = features
data['Non_Severe'] = targets['Non_Severe'].values
data
/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  
Fever	Tiredness	Dry-Cough	Difficulty-in-Breathing	Sore-Throat	Pains	Nasal-Congestion	Runny-Nose	Diarrhea	Age_0-9	Age_10-19	Age_20-24	Age_25-59	Age_60+	Gender_Male	Gender_Female	Gender_Transgender	Total_Symptom	Non_Severe
0	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0	11	0
1	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0	11	0
2	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0	11	0
3	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0	11	0
4	1	1	1	1	1	1	1	1	1	1	0	0	0	0	1	0	0	11	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
316795	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	1	2	0
316796	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	1	2	0
316797	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	1	2	1
316798	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	1	2	1
316799	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	1	2	1
316800 rows × 19 columns

data_for_corr = data.drop(labels="Total_Symptom", axis=1)
# data_for_corr['Condition'] = data_for_corr['Condition'].apply(make_condition_grade)
corrmat = data_for_corr.corr()
k = 22
cols = corrmat.nlargest(k, 'Non_Severe')['Non_Severe'].index
cm = np.corrcoef(data_for_corr[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

Modelling
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
x = data.drop(['Non_Severe', 'Total_Symptom'], axis=1)
x = PCA(n_components = 3).fit_transform(x)
y = data['Non_Severe']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.3)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc.score(x_test, y_test)
0.7509574915824916
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr.score(x_test, y_test)
0.7509574915824916
DTC = DecisionTreeClassifier()
DTC.fit(x_train, y_train)
DTC.score(x_test, y_test)
0.7509574915824916
params = {
    "max_depth":[15,20,25], 
    "n_estimators":[27,30,33],
    "criterion":["gini", "entropy"],
}

rfc = RandomForestClassifier()

rf_reg = GridSearchCV(rfc, params, cv = 10, n_jobs =10)
rf_reg.fit(x_train, y_train)
print(rf_reg.best_estimator_)
RandomForestClassifier(max_depth=15, n_estimators=27)
rfc_tune = RandomForestClassifier(max_depth=15, n_estimators=27)
rfc_tune.fit(x_train, y_train)
score = cross_val_score(rfc,x_test,y_test,cv = k_fold,n_jobs=1,scoring="accuracy")
print(score.mean())
0.7507154882154883
params={
    "penalty":['l1', 'l2', 'elasticnet', 'none'],
    "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
lr = LogisticRegression()
lr_reg = GridSearchCV(lr, params, cv=10, n_jobs=10)
lr_reg.fit(x_train, y_train)
print(lr_reg.best_estimator_)
/opt/conda/lib/python3.7/site-packages/sklearn/model_selection/_search.py:921: UserWarning: One or more of the test scores are non-finite: [       nan        nan 0.74958965        nan 0.74958965 0.74958965
 0.74958965 0.74958965 0.74958965 0.74958965        nan        nan
        nan        nan        nan 0.74958965 0.74958965        nan
 0.74958965 0.74958965]
  category=UserWarning
LogisticRegression(penalty='l1', solver='liblinear')
lr_tune = LogisticRegression(penalty='l1', solver='liblinear')
lr_tune.fit(x_train, y_train)
score = cross_val_score(lr_tune, x_test, y_test, cv=k_fold, n_jobs=1, scoring="accuracy")
print(score.mean())
0.7509574915824916
params = {
    "criterion":["gini", "entropy"],
    "max_depth":[15,20,25], 
}
dtc = DecisionTreeClassifier()
dtc_reg = GridSearchCV(dtc, params, cv=10, n_jobs=10)
dtc_reg.fit(x_train, y_train)
print(dtc_reg.best_estimator_)
DecisionTreeClassifier(max_depth=15)
dtc_tune = DecisionTreeClassifier(max_depth=15)
dtc_tune.fit(x_train, y_train)
score = cross_val_score(dtc_tune, x_test, y_test, cv=k_fold, n_jobs=1, scoring="accuracy")
print(score.mean())
0.7508838383838384
