from keras.models import load_model
import numpy as np
#load the model
mod = load_model('grade.h5')
lst = []
#input the number to predict
in1 = float(input('how many hours of studying:'))
in2 = float(input('what was ur last exam score:'))
#normolize the hours study and the preivious exam
in1 = in1/24.0
in2 = in2/100.0
lst.append(in1)
lst.append(in2)
lst = np.array(lst,dtype='float32').reshape(1,2) 
#make numpy array to predict and change the shape to the net
out = mod.predict(lst)
#predict the pass or fail 
arg = np.argmax(out)
print(arg)