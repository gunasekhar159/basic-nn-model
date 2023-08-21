##EXP1: Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type. You can train the model by providing the model and the tagged dataset as an input to Train Model.

In this experiment we need to develop a Neural Network Regression Model so first we need to create a dataset (i.e : an excel file with some inputs as well as corresponding outputs).Then upload the sheet to drive then using corresponding code open the sheet and then import the required python libraries for porcessing.

Use df.head to get the first 5 values from the dataset or sheet.Then assign x and y values for input and coressponding outputs.Then split the dataset into testing and training,fit the training set and for the model use the "relu" activation function for the hidden layers of the neural network (here two hidden layers of 4 and 6 neurons are taken to process).To check the loss mean square error is uesd.Then the testing set is taken and fitted, at last the model is checked for accuracy via preiction.


## Neural Network Model

![img 1](https://github.com/Ramsai1234/basic-nn-model/assets/94269989/185ca9b4-a5c0-4c43-ba71-6cd6d2c64368)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Developed by: M.Gunasekhar
Reg no : 212221240014
```
```
### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('model 1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])
df = df.astype({'input':'int','output':'int'})

### Import the packages :

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df.head()

X = df[['input']].values
y = df[['output']].values
X

### Split Training and testing set :

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)

### Pre-processing the data :

Scaler = MinMaxScaler()
Scaler.fit(X_train)
Scaler.fit(X_test)

X_train1 = Scaler.transform(X_train)
X_test1 = Scaler.transform(X_test)

X_train1

### Model :

ai_brain = Sequential([
    Dense(4,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1,y_train,epochs = 100)

### Loss plot :

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

### Testing with the test data and predicting the output :

ai_brain.evaluate(X_test1,y_test)

X_n1 = [[38]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
```

## Dataset Information
<img width="299" alt="2" src="https://github.com/gunasekhar159/basic-nn-model/assets/95043391/630d132a-07b6-4935-948c-1e9d8f1c1e47">





## OUTPUT

### Training Loss Vs Iteration Plot

<img width="453" alt="3" src="https://github.com/gunasekhar159/basic-nn-model/assets/95043391/4a30129c-55fc-4c13-b04f-ffc1aed18bdc">




### Test Data Root Mean Squared Error
<img width="453" alt="3" src="https://github.com/gunasekhar159/basic-nn-model/assets/95043391/7b9c861c-8b57-48b5-91ff-2f329f5dca80">



### New Sample Data Prediction
<img width="433" alt="1" src="https://github.com/gunasekhar159/basic-nn-model/assets/95043391/1a2a5782-096a-42d1-a96e-abe6d44c92aa">




## RESULT
Thus a neural network model for regression using the given dataset is written and executed successfully.


