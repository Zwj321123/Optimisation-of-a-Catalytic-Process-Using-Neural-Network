import numpy as np

from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model
from numpy import asarray
from numpy import savetxt
import os
#Extracct the data
X = np.genfromtxt('Input.csv',delimiter=',')
y = np.genfromtxt('Output2.csv',delimiter=',')
xPre = np.genfromtxt('input_Pre.csv', delimiter = ',')

#Normalization
X_scale = preprocessing.scale(X)
#print (X_scale)
X_PreScale = preprocessing.scale(xPre)
#X_train = X_scale[:19]
X_train = X_scale;
#X_test = X_scale[19:]
X_test = X_PreScale;
#y_train_2 = y[:15,2]
y_train_2 = y[:,:3]
#y_test= y[19:,:3]

#y_test_2 = y[15:19,2]

n_hidden_1 = 12#neurons of hidden layer 1
n_hidden_2 = 12#neurons of hidden layer 2
n_input = 3 #number of input (4 in this project)
n_output= 3
training_epochs = 2000 # epochs
batch_size = 15#number of batch



model = Sequential()
model.add(Dense(n_hidden_1, activation='relu', input_dim=n_input,kernel_regularizer=regularizers.l2(0.01))) #Regularizer
model.add(BatchNormalization())
model.add(Dropout(0.2)) #Dropout
#model.add(Dense(n_hidden_2, activation='relu'))

model.add(BatchNormalization())#Normalization
model.add(Dense(n_output))
model.compile(loss='mean_squared_error', optimizer='adam')
model.save('my_model.h5')
history = model.fit(X_train, y_train_2, batch_size=batch_size, epochs=training_epochs, validation_data = (X_train, y_train_2))
#predict
pred_test_y = model.predict(X_test)
#model.save('my_model.h5')
'''
r=[0,0,0]
predict = model.predict(X_test)
print (predict)
data = asarray(predict)
i = 0
print(model.get_config())
for layer in model.layers:
    weights = layer.get_weights()

    print(i)
    print(weights)
    if (i==0):
        savetxt('weights1.csv', weights[0], delimiter=',')
        os.startfile('weights1.csv')
    elif (i == 4):
        savetxt('weights2.csv', weights[0], delimiter=',')
        os.startfile('weights2.csv')
    i=i+1
'''
# save numpy array as csv file
savetxt('predictOutput2.csv', pred_test_y, delimiter=',')
os.startfile('predictOutput2.csv')
'''
print (pred_test_y)
a = 0

for i in range (np.size(pred_test_y, 0)):#np.size(predict, 0)
    for j in range (np.size(pred_test_y, 1)):#np.size(predict, 1)
        r[j]+=abs((pred_test_y[i,j]-y_test[i,j])/y_test[i,j])
for i in r:
    a+=(i/np.size(pred_test_y, 0))
print(a/3)
pred_acc = r2_score(y_test, pred_test_y)
print('pred_acc',pred_acc)
'''
