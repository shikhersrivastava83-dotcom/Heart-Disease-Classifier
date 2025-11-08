import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time as t
from sklearn.model_selection import cross_val_score
RANDOM_STATE=55

#TRAINING DATASETS
df_old = pd.read_csv('heart.csv')
df_new = pd.read_csv('heart_disease_uci.csv')
df1 = pd.read_csv('Cardiovascular_Disease_Dataset.csv')
df_new = df_new.drop(['ca', 'thal', 'id', 'dataset'], axis=1)
df1 = df1.drop(['noofmajorvessels', 'patientid'], axis=1)

df2 = pd.read_csv('Heart_disease_statlog.csv')
df2 = df2.drop(['ca', 'thal'], axis=1)
df2['ST_Slope'] = df2['ST_Slope'].replace({0: 1, 1: 2, 2: 3})   #Aligning to same values

# Data Cleaning - normalize values across both datasets
mapping_cp = {
    'TA': 'TA', 'ASY': 'ASY', 'ATA': 'ATA', 'NAP': 'NAP',
    'typical angina': 'TA', 'asymptomatic': 'ASY',
    'non-anginal': 'NAP','atypical angina': 'ATA',
    0 : 'TA', 1: 'ATA', 2:'NAP', 3: 'ASY'
}
mapping_ecg = {
    'Normal': 'Normal', 'LVH': 'LVH', 'ST': 'ST',
    'lv hypertrophy': 'LVH','normal': 'Normal','st-t abnormality': 'ST',
    0 : 'Normal', 1: 'ST', 2: 'LVH'
}
mapping_slope = {
    'Down': 'Down', 'Up': 'Up', 'Flat': 'Flat',
    'downsloping': 'Down','flat': 'Flat','upsloping': 'Up',
    1: 'Up', 2: 'Flat', 3: 'Down'
}

# concatenate first so we can normalize all rows in the same way
df = pd.concat([df_old, df_new, df1, df2], ignore_index=True)

# helper to convert many possible representations to 0/1
def to_binary_01(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, bool):
        return int(val)
    s = str(val).strip().upper()
    if s in ('Y','YES','1','TRUE','T'):
        return 1
    if s in ('N','NO','0','FALSE','F'):
        return 0
    try:
        return int(float(val))
    except:
        return np.nan

df['ST_Slope'] = df['ST_Slope'].replace(0, np.nan)
df['ChestPainType'] = df['ChestPainType'].replace(mapping_cp)
df['RestingECG'] = df['RestingECG'].replace(mapping_ecg)
df['ST_Slope'] = df['ST_Slope'].replace(mapping_slope)
df['Sex'] = df['Sex'].replace({'Male': 'M', 'Female': 'F'})


# normalize ExerciseAngina and FastingBS to numeric 0/1
if 'ExerciseAngina' in df.columns:
    df['ExerciseAngina'] = df['ExerciseAngina'].apply(to_binary_01)

if 'FastingBS' in df.columns:
    df['FastingBS'] = df['FastingBS'].apply(to_binary_01)

# ensure HeartDisease is binary 0/1 (some datasets have >0 as disease)
if 'HeartDisease' in df.columns:
    df['HeartDisease'] = df['HeartDisease'].apply(lambda x: 1 if (pd.notna(x) and float(x) > 0) else 0)


#splliting
x = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
x_, x_train, y_, y_train = train_test_split(x, y, test_size=0.8, random_state=RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_, y_, test_size=0.5, random_state=42)


#OUTLIERS
SEE_OUTLIERS_TRAIN = False         #SET IT TO TRUE TO SEE OUTLIERS in training set
    #outliers in training set
if SEE_OUTLIERS_TRAIN:                             
    for col in x_train.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(6, 4))
        plt.boxplot(x_train[col].dropna())
        plt.title(f"{col} Boxplot IN TRAIN")
        plt.show()

col = ['RestingBP', 'Cholesterol', 'Oldpeak', 'MaxHR']
Q1 = x_train[col].quantile(0.25)
Q3 = x_train[col].quantile(0.75)
IQR = Q3 - Q1
for i in col:
    x_train[i] = x_train[i].clip(lower=Q1[i] - 1.5 * IQR[i], upper=Q3[i] + 1.5 * IQR[i])

        #outlers in validation set

SEE_OUTLIERS_VAL = False     #SET IT TO TRUE TO SEE OUTLIERS in validation set
if SEE_OUTLIERS_VAL:
    for columns in x_val.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(6, 4))
        plt.boxplot(x_val[columns].dropna())
        plt.title(f"{columns} Boxplot IN VAL")
        plt.show()
for i in col:
    x_val[i] = x_val[i].clip(lower=Q1[i] - 1.5 * IQR[i], upper=Q3[i] + 1.5 * IQR[i])

        #killing outliers in test set

for i in col:
    x_test[i] = x_test[i].clip(lower=Q1[i] - 1.5 * IQR[i], upper=Q3[i] + 1.5 * IQR[i])

#filling null values

num_col = ['Age', 'Cholesterol', 'RestingBP', 'MaxHR', 'Oldpeak']
for col in num_col:
    x_train[col] = x_train[col].fillna(x_train[col].median())
    x_val[col] = x_val[col].fillna(x_train[col].median())      #median is robust
    x_test[col] = x_test[col].fillna(x_train[col].median())

cat_col = ['ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for i in cat_col:
    x_train[i] = x_train[i].fillna(x_train[i].mode()[0])
    x_val[i] = x_val[i].fillna(x_train[i].mode()[0])
    x_test[i] = x_test[i].fillna(x_train[i].mode()[0])



#Encoding

x_train['Sex'] = x_train['Sex'].map({'M': 1, 'F': 0})
x_val['Sex'] = x_val['Sex'].map({'M': 1, 'F': 0})
x_test['Sex'] = x_test['Sex'].map({'M': 1, 'F': 0})

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first', dtype=int)

x_train_encoded = encoder.fit_transform(x_train[['ChestPainType', 'RestingECG', 'ST_Slope']])
x_val_encoded = encoder.transform(x_val[['ChestPainType', 'RestingECG', 'ST_Slope']])
x_test_encoded = encoder.transform(x_test[['ChestPainType', 'RestingECG', 'ST_Slope']])

x_train_encoded_df = pd.DataFrame(x_train_encoded, columns=encoder.get_feature_names_out(['ChestPainType', 'RestingECG', 'ST_Slope']))
x_val_encoded_df = pd.DataFrame(np.array(x_val_encoded), columns=encoder.get_feature_names_out(['ChestPainType', 'RestingECG', 'ST_Slope']))
x_test_encoded_df = pd.DataFrame(np.array(x_test_encoded), columns=encoder.get_feature_names_out(['ChestPainType', 'RestingECG', 'ST_Slope']))

x_train_encoded_df.index = x_train.index
x_val_encoded_df.index = x_val.index
x_test_encoded_df.index = x_test.index

x_train = pd.concat([x_train, x_train_encoded_df], axis=1)
x_val = pd.concat([x_val, x_val_encoded_df], axis=1)
x_test = pd.concat([x_test, x_test_encoded_df], axis=1)

x_train = x_train.drop(['ChestPainType', 'RestingECG', 'ST_Slope'], axis=1)
x_val = x_val.drop(['ChestPainType', 'RestingECG', 'ST_Slope'], axis=1)
x_test = x_test.drop(['ChestPainType', 'RestingECG', 'ST_Slope'], axis=1)

x_train_pd = x_train


#training set, cross validation set and test set
x_train = x_train.values
x_val = x_val.values  
y_train = y_train.values
y_val = y_val.values
x_test = x_test.values
y_test = y_test.values

#-------------------------------------DECISION TREE----------------------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-~-~-~-~-~-~-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

min_samples_split_list = [10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [2, 3, 4, 8, 16, 32, 64, None]

best_acc = 0
best_params ={}
for depth in [2, 3, 4, 5, 6, 7, 8, 9, None]:
    for min_sample_splits in min_samples_split_list:
        for max in [0.22, 0.4, 0.8, 0.9, 'log2', 'sqrt']:
            tree = DecisionTreeClassifier(min_samples_split=min_sample_splits, max_depth=depth, max_features=max, random_state=55).fit(x_train, y_train)
            prediction_train = tree.predict(x_train)
            prediction_val = tree.predict(x_val)
            accuracy_train = accuracy_score(y_train, prediction_train)
            accuracy_val = accuracy_score(y_val, prediction_val)
            if accuracy_val > best_acc:
                best_acc = accuracy_val
                best_params = {
                    'min_samples_split': min_sample_splits,
                    'max_depth': depth,
                    'max_features': max
                }
print(f"Highest accuracy in val: {best_acc *100:.2f}%")
print(best_params)
tree = DecisionTreeClassifier(**best_params, random_state=55, min_samples_leaf=1).fit(x_train, y_train)  #keep the random state same for consistency
ypred_train = tree.predict(x_train)
ypred_val = tree.predict(x_val)
ypred_test = tree.predict(x_test)


print(f"Acc for training set: {accuracy_score(y_train, ypred_train) *100:.2f}%\nAcc for val: {accuracy_score(y_val, ypred_val) * 100:.2f}%")
print(f"Acc for test set: {accuracy_score(y_test, ypred_test) *100:.2f}%")
fi = pd.Series(tree.feature_importances_, index=x_train_pd.columns).sort_values(ascending=False)

print(f"-------------------------------------------------------")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RANDOM_FOREST~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


n_estimators_list = [50, 100, 200, 500]
min_samples_split_list_rf = [10, 30, 50, 100, 200, 300]


rf_acc = 0
best_rf_params = {}

TUNING = not True    #SET IT TO TRUE TO SEE THE BEST COMBO MARK! IT MAY TAKE few minutes...
if TUNING:
    start = t.time()
    for n in n_estimators_list:
        for split in min_samples_split_list_rf:
            for max_depth in [3, 4, 5, 6 , 7, 9, None]:
                for leaf in [5, 10, 20, 30]:
                    rf = RandomForestClassifier(n_estimators=n, min_samples_split=split, min_samples_leaf=leaf, max_depth=max_depth, random_state=55).fit(x_train, y_train)
                    yhattrain_rf = rf.predict(x_train)
                    yhatval_rf = rf.predict(x_val)
                    acc_val_rf = accuracy_score(yhatval_rf, y_val)
                    if acc_val_rf > rf_acc:
                        rf_acc = acc_val_rf
                        best_rf_params = {'n_estimators': n,
                                          'min_samples_split': split,
                                          'max_depth': max_depth,
                                          'min_samples_leaf': leaf
                                        }
    print(f"highest val accuracy obtained: {rf_acc * 100:.2f}%")
    print(best_rf_params)
    end = t.time()
    print(f"time taken : {(end - start)/60} minutes")




#THE COMBO OF HYPERPARAMETERS ARE CHOSEN THROUGH NESTED LOOPS TO GE THE BEST COMBO 

#RANDOM FOREST ON BEST HYPER PARAMETERS
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, max_depth=None, min_samples_leaf=5, random_state=42).fit(x_train, y_train)
y_pred_train_rf = rf.predict(x_train)
y_pred_val_rf = rf.predict(x_val)
y_pred_test_rf = rf.predict(x_test)

print(f"random forest on training with best hyperpara.: {accuracy_score(y_train, y_pred_train_rf) * 100:.2f}%")
print(f"On validation set: {accuracy_score(y_val, y_pred_val_rf) * 100:.2f}%")
print(f"On test set: {accuracy_score(y_test, y_pred_test_rf) * 100:.2f}%")
