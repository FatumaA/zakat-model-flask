import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib


# Read csv and put in Dataframe
f = pd.read_csv('csv2zakatmixed.csv')
df = pd.DataFrame(f)

# Drop columns with >70% NaN values
perc = 70.0 
min_count =  int(((100-perc)/100)*df.shape[0] + 1)
df = df.dropna( axis=1, 
                thresh=min_count)

# Replace NA with Not applicable and bundle low frequency categories into 'Secondary and university'
df['Level of education'] = df['Level of education'].fillna('Not Applicable')

top_two_levelofeducation = df['Level of education'].value_counts().nlargest(2).index

df['Level of education'] = df['Level of education'].where(df['Level of education'].isin(top_two_levelofeducation), other = 'Secondary and University')

# 
# 
# Turn no. of kids to ranges
bins = [0, 1, 3, 6, np.inf]
names= ['0', '1-3', '3-6', 'Over 6']

df['No. of children'] = pd.cut(df['No. of children'], bins, include_lowest = True, labels = names)


# Turn no, of dependants to ranges
df['no. of other dependents'] = df['no. of other dependents'].fillna(0)
bins = [0, 1, 3, np.inf]
names= ['0', '1-3', 'Over 3']

df['no. of other dependents'] = pd.cut(df['no. of other dependents'], bins, include_lowest = True, labels = names)


# Treat remaining null values
df['Your income '] = df['Your income '].fillna(0)
df['Monthly expenses'] = df['Monthly expenses'].fillna(0)


# Turn categorical data to numeric
df['Gender'] = pd.get_dummies(df['Gender'])

df = pd.get_dummies(df, columns=['Level of education', 'No. of children', 'no. of other dependents', 'Have you applied to other organizations?', 'Are you disabled?', 
                                    'Is first application?', 'Marital Status', 'Given'], drop_first = True)


# 
# 
# Declare features and label
# # Features
X = df.iloc[:, [0,14]].values
# Label
y = df.iloc[:, 15].values


# 
# Split our data
x_train, x_test,y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
# Initialize our classifier
model = KNeighborsClassifier(n_neighbors=3)

# Train our classifier
model.fit(x_train, y_train)

# Save model
joblib.dump(model, 'zakatfinalized_model.sav')

# load saved model
loadedModel = joblib.load('zakatfinalized_model.sav')

model_columns = list(X)
joblib.dump(model_columns, 'model_columns.pkl')
result = loadedModel.score(x_test, y_test)
# predictions = model.predict(x_test)

print("Accuracy:",result)
