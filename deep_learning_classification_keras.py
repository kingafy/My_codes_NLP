# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:32:00 2020

@author: anshu
"""

###deep_learning
import numpy as np
import keras
import os
os.chdir(r"D:\Data science\NLP_learnings\jose portella\UPDATED_NLP_COURSE\06-Deep-Learning")

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
from keras.utils import to_categorical

y = to_categorical(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train
X_test

from sklearn.preprocessing import MinMaxScaler

scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)
X_train.max()

scaled_X_train.max()
X_train

##building neural networks with keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(scaled_X_train,y_train,epochs=150, verbose=2)

model.predict_classes(scaled_X_test)

model.metrics_names
###evaluation on Test data set
model.evaluate(x=scaled_X_test,y=y_test)
from sklearn.metrics import confusion_matrix,classification_report
predictions = model.predict_classes(scaled_X_test)

predictions
y_test.argmax(axis=1)
confusion_matrix(y_test.argmax(axis=1),predictions)

print(classification_report(y_test.argmax(axis=1),predictions))


model.save('myfirstmodel.h5')
from keras.models import load_model

newmodel = load_model('myfirstmodel.h5')
newmodel.predict_classes(X_test)