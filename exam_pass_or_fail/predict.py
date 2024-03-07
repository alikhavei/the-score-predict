import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers,models
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

plt.style.use("ggplot")
#read the csv file with pandas
x = pd.read_csv('student_exam_data.csv')
#make dataframe of the csv
u = pd.DataFrame(x)
#convert Data frame to dictionary
u = u.to_dict()
#split the all number in column to diffrent  variable
study = u['Study Hours']
pv_exam = u['Previous Exam Score']
pass_or_fail = u['Pass/Fail']
lst1 = []
lst_target = []
#loop through the  study hours and previous  exam score  to convert it to tuple
for i in range(len(study)):
    r= float(study[i])
    r1 = float(pv_exam[i])
    passes = pass_or_fail[i]
    #normalize the data
    r = r/24.0
    r1 = r1/100.0
    r= str(r)
    r1 = str(r1)
    r = r.split()
    r1 = r1.split()
    t = r+r1
    #put the data in one tuple and append it to the list outside the loop
    t = tuple(t)
    lst1.append(t)
    lst_target.append(int(passes))
#make the target to numpy array to encode it
lst_target = np.array(lst_target)
#encoding the target
l_e = LabelEncoder()
int_incoding = l_e.fit_transform(lst_target)
one_hot_encoding = to_categorical(int_incoding)
#convert the list that contains tuple to numpy array to the net
lst1 = np.array(lst1,dtype='float32')
x_train , x_test , y_train , y_test = train_test_split(lst1,one_hot_encoding,test_size=0.2)
#making the net
net = models.Sequential([layers.Flatten(),  
                         layers.Dense(64,activation='relu'),
                         layers.BatchNormalization(),
                         layers.Dropout(0.2),
                         layers.Dense(32,activation='relu'),
                         layers.BatchNormalization(),
                         layers.Dense(2,activation='softmax')
])

net.compile(optimizer='Adam',metrics=['accuracy'],loss='categorical_crossentropy')
n = net.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=15)

plt.plot(n.history['accuracy'], label = 'train accuracy')
plt.plot(n.history['val_accuracy'], label = 'test accuracy')

plt.plot(n.history['loss'], label = 'train loss')
plt.plot(n.history['val_loss'], label = 'test loss')
plt.legend(loc='best')
plt.show()
#save the net
net.save('grade.h5')
