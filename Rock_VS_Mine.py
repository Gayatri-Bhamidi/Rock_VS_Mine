import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading dataset
sonar_data = pd.read_csv('/content/sonar data.csv', header = None)

sonar_data.head()

# number of rows and scolumns
sonar_data.shape

# statistical measures of the data
sonar_data.describe()

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

# seperating data and variables
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)

print(x.shape, x_train.shape, x_test.shape)

print(x_train)
print(y_train)

model = LogisticRegression()
# training the logistic regression model with training data
model.fit(x_train, y_train)
print(model)

# accuracy on training data
x_train_predection = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predection, y_train)

print('Accuracy on training data : ', training_data_accuracy)

# accuracy on test data
x_test_predection = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_predection, y_test)

print('Accuracy on testing data : ', testing_data_accuracy)

input_data = (0.0124,0.0433,0.0604,0.0449,0.0597,0.0355,0.0531,0.0343,0.1052,0.2120,0.1640,0.1901,0.3026,0.2019,0.0592,0.2390,0.3657,0.3809,0.5929,0.6299,0.5801,0.4574,0.4449,0.3691,0.6446,0.8940,0.8978,0.4980,0.3333,0.2350,0.1553,0.3666,0.4340,0.3082,0.3024,0.4109,0.5501,0.4129,0.5499,0.5018,0.3132,0.2802,0.2351,0.2298,0.1155,0.0724,0.0621,0.0318,0.0450,0.0167,0.0078,0.0083,0.0057,0.0174,0.0188,0.0054,0.0114,0.0196,0.0147,0.0062)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape np array as we predecting for only onevalue
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

predection = model.predict(input_data_reshaped)
print(predection)

if(predection[0] == 'R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')
  
  
input_data = (0.0129,0.0141,0.0309,0.0375,0.0767,0.0787,0.0662,0.1108,0.1777,0.2245,0.2431,0.3134,0.3206,0.2917,0.2249,0.2347,0.2143,0.2939,0.4898,0.6127,0.7531,0.7718,0.7432,0.8673,0.9308,0.9836,1.0000,0.9595,0.8722,0.6862,0.4901,0.3280,0.3115,0.1969,0.1019,0.0317,0.0756,0.0907,0.1066,0.1380,0.0665,0.1475,0.2470,0.2788,0.2709,0.2283,0.1818,0.1185,0.0546,0.0219,0.0204,0.0124,0.0093,0.0072,0.0019,0.0027,0.0054,0.0017,0.0024,0.0029)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape np array as we predecting for only onevalue
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

predection = model.predict(input_data_reshaped)
print(predection)

if(predection[0] == 'R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')
  
  
