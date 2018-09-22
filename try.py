from __future__ import division
import logging
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid
from const import ModelChoice, CITY_BLOCK_DICT, FeatureChoice, ReducerChoice, PATH_PATTERN, TARGET, FEATURE_DICT
from const import LOG_DIR, ScaleChoice
import pywt


def dwt_loss(y_true, y_pred):
	print(y_true)
	print(y_pred)
	print(tf.rank(y_true))
	print(tf.rank(y_pred))
	#raw_input("\n")
	return K.mean(abs(y_pred - y_true) , axis=-1) 


train_dfs = [pd.read_csv(PATH_PATTERN)]
train_df = pd.concat(train_dfs)

test_dfs = [pd.read_csv(PATH_PATTERN)]
test_df = pd.concat(test_dfs)

x_train = train_df[FEATURE_DICT[FeatureChoice.all]].values
y_train = train_df[TARGET].values

x_test = test_df[FEATURE_DICT[FeatureChoice.all]].values
y_test = test_df[TARGET].values

check = test_df['loss'].values

reducer = PCA(n_components=50)
estimator = [ ('scaler', StandardScaler()),('reducer', reducer)]

pipe = Pipeline(estimator)
x = pipe.fit_transform(np.vstack([x_train, x_test]))
split_index = len(x_train)
x_train = x[:split_index, :]
x_test = x[split_index:, :]

print(np.shape(x_train))

X = []
Y = []
x_test = []
y_test = []
cnt = 0
for k in range(len(x_train)):
	i = k % 61
	j = k // 61
	if (i >= 2 and i <= 58):
		if (j >= 2 and j <= 54 and check[k] < 0.25):
			cnt = cnt + 1
			Z = []
			for yy in range(7):
				ZZ = []
				for xx in range(7):
					ZZ.append(x_train[(i + xx - 2) + (j + yy - 2) * 61])
				Z.append(ZZ)
			X.append(Z)
			cA , cD = pywt.dwt(y_train[(i) + (j) * 61] , 'db6')
			Y.append(y_train[(i) + (j) * 61])
X = np.array(X)
#print(X)
print(np.shape(X))
Y = np.array(Y)
#print(Y)
print(np.shape(Y))


x_train,x_test,y1,y2 = train_test_split(X,Y,test_size=0.1,random_state=0)
y_train = []
y_test = []
for i in range(len(y1)):
	cA , cD = pywt.dwt(y1[i] , 'db6')
	kk = 0
	for j in range(29):
		kk = kk + cA[j]
		#if (cA[j] > kk):
		#	kk = cA[j]
	cA = cA / kk
	#y_train.append(y1[i])
	y_train.append([cA[0],cA[1],cA[2],cA[3],cA[4],cA[5],cA[6],cA[7],cA[8],cA[9],cA[10],cA[11],cA[12],cA[13],cA[14],cA[15],cA[16],cA[17],cA[18],cA[19],cA[20],cA[21],cA[22],cA[23],cA[24],cA[25],cA[26],cA[27],cA[28]])
for i in range(len(y2)):
	cA , cD = pywt.dwt(y2[i] , 'db6')
	kk = 0
	for j in range(29):
		kk = kk + cA[j]
		#if (cA[j] > kk):
		#	kk = cA[j]
	cA = cA / kk
	#y_test.append(y2[i])
	y_test.append([cA[0],cA[1],cA[2],cA[3],cA[4],cA[5],cA[6],cA[7],cA[8],cA[9],cA[10],cA[11],cA[12],cA[13],cA[14],cA[15],cA[16],cA[17],cA[18],cA[19],cA[20],cA[21],cA[22],cA[23],cA[24],cA[25],cA[26],cA[27],cA[28]])

print(np.shape(x_train))
print(np.shape(y_train))
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,MaxPooling2D
from keras.models import Sequential 
from keras import backend as K  
K.set_image_dim_ordering('tf') 
model = Sequential()
model.add(Conv2D(16,
                 activation='relu',
                 input_shape=(7 , 7 , 50),
                 padding='valid',
                 nb_row=1,
                 nb_col=1))
#model.add(Dropout(0.2))
model.add(Conv2D(32, activation='relu',
				 padding='valid',
                 nb_row=3,
                 nb_col=3))
#model.add(Dropout(0.2))
model.add(Conv2D(16, activation='relu',
				 padding='valid',
                 nb_row=1,
                 nb_col=1))
#model.add(Dropout(0.2))
model.add(Conv2D(32, activation='relu',
				 padding='valid',
                 nb_row=3,
                 nb_col=3))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(29))
optimizer = Adam(lr = 0.00001)
model.compile(optimizer=optimizer, loss=dwt_loss)
minn = 10000
k = 10000
mink = []
mino = []
minans = []
my_ans = []
print(np.shape(x_train))
#print((x_train))
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)
print(np.shape(y_train))
#print((y_train))
maxe = 0
my_anss = []
for e in range(2):
	model.fit(x_train , y_train,batch_size=32,epochs=1,validation_data=[x_test , y_test])
	ans = model.predict(x_test)
	#print(ans)
	#raw_input("\n")
	#print(x_test)
	A = []
	anss = 0
	n = 0
	for i in range(len(ans)):
		X = pywt.idwt(ans[i] , None , 'db6')
		#X = ans[i]
		kk = 0
		for j in range(48):
			kk = kk + X[j]
			#if (X[j] > kk):
			#	kk = X[j]
		X = X / kk

		Y = y2[i]
		kk = 0
		for j in range(48):
			kk = kk + Y[j]
		Y = Y / kk
		if (np.linalg.norm((X - Y) , ord = 2) / np.linalg.norm((Y) , ord = 2)  > 0):
			anss = anss + np.linalg.norm((X - Y) , ord = 2) / np.linalg.norm((Y) , ord = 2) 
			n = n + 1
		A.append(np.linalg.norm((X - Y) , ord = 2) / np.linalg.norm((Y) , ord = 2))
		if (np.linalg.norm((X - Y) , ord = 2) / np.linalg.norm((Y) , ord = 2) < minn):
			minn = np.linalg.norm((X - Y) , ord = 2) / np.linalg.norm((Y) , ord = 2)
			mink = ans[i]
			mino = y2[i]
			maxe = e


	#print(k)
	#print(minn)
	#print(ans[k])
	#print(y_test[k])
	print(anss / n)
	if (anss / n < k):
		k = anss / n
		minans = A
		my_ans = []
		my_anss = []
		for i in range(len(ans)):
			my_ans.append(pywt.idwt(ans[i] , None , 'db6'))
		ans = model.predict(x_train)
		for i in range(len(ans)):
			my_anss.append(pywt.idwt(ans[i] , None , 'db6'))
#minans.sort()

print(k)
print(minans)
print(minn)
print(mink)
print(mino)
print(maxe)
z = np.vstack((my_ans))

import pandas
output = pandas.DataFrame(list(z))
output.to_csv("my_ans.csv", index=False)

z = np.vstack((y_test))

output = pandas.DataFrame(list(z))
output.to_csv("my_test.csv", index=False)

z = np.vstack((y1))

output = pandas.DataFrame(list(z))
output.to_csv("my_train.csv", index=False)

z = np.vstack((my_anss))

output = pandas.DataFrame(list(z))
output.to_csv("my_train_ans.csv", index=False)