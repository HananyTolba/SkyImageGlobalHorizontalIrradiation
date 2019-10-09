# %%
import scipy.io as sio

# %%
import os

# %%
mat_contents = sio.loadmat('ImGHIrec2.mat')

# %%
path=os.getcwd()

# %%


# %%
DateTimeCom=sio.loadmat('DateTimeCom.mat')
dt=DateTimeCom['DateTimeCom']


# %%
import numpy as np




# %%
import pandas as pd
dt=pd.to_datetime(dt)

# %%
mat_contents.keys()

# %%
mat_Im=mat_contents["ImGHIrec2"]
mat_Im.shape

# %%
x=mat_Im[:,:,70]
y=mat_Im[:,:,75]

# %%
x=x.reshape(x.size,1)
y=y.reshape(y.size,1)



# %%

from matplotlib import pyplot as plt

# %%
ghi=mat_contents["GHI"]
ghiset=ghi[:13000]
ghiset_orig=ghiset

max(ghiset)

# %%
dataset_orig=mat_Im.reshape(mat_Im.shape[0]*mat_Im.shape[1],mat_Im.shape[2])

dataset_orig=dataset_orig[:,:13000]


#dataset_orig=dataset_orig[:,::4]




from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset_orig)

ghiset=scaler.fit_transform(ghiset)

#df_test['scaled_GHI'] = scaler.fit_transform(np.array(df_test['GHI']).reshape(-1, 1))

train_size=int(dataset_orig.shape[1]*.85)

data_train=dataset[:,:train_size]
data_test=dataset[:,train_size:]
data_train=data_train.T
data_test=data_test.T
ghiset_train=ghiset[:train_size]
ghiset_test=ghiset[train_size:]

# %%
data_train.shape,data_test.shape,dataset_orig.shape

#train_size

# %%


# %%
dimInput=1
X_train, y_train = data_train,ghiset_train
X_val, y_val = data_test,ghiset_test

# %%
X_val.shape,y_val.shape,X_train.shape,y_train.shape



from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


#definition des entrées. None indicates the number of instances
input_layer = Input(shape=(728,), dtype='float32')

#Les couches Denses avec une fonction d'activation relu
dense1 = Dense(128, activation='sigmoid')(input_layer)
dropout_layer = Dropout(0.1)(dense1)


dense2 = Dense(128, activation='sigmoid')(dense1)
dropout_layer = Dropout(0.1)(dense2)


dense3 = Dense(64, activation='sigmoid')(dense2)
dropout_layer = Dropout(0.1)(dense3)

#dense4 = Dense(8, activation='relu')(dense3)
#dropout_layer = Dropout(0.1)(dense4)



output_layer = Dense(1, activation='sigmoid')(dropout_layer)




ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_squared_error', optimizer='adam')

ts_model.summary()






newest = "CamGHI_MLP_estim.hdf5"
print(newest)

# %%
os.path.join(path,newest)

# %%
best_model = load_model(os.path.join(path,newest))
preds = best_model.predict(X_val)

ghi_est = scaler.inverse_transform(preds)
#ghi_est = np.squeeze(ghi_est)

y_val=scaler.inverse_transform(y_val)

from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_val, ghi_est)

#print('R-squared for the validation set:'.format(r2))
print("R-squared en test  est {} ".format(r2))

# calculate root mean squared error
testScore = np.sqrt(mean_squared_error(y_val, ghi_est))
print("L'erreur en test  est {} (nRMSE)".format(testScore))



# %%
#ghi_est=pred_PRES

ghi_true=y_val
plt.figure
plt.plot(ghi_true)
plt.plot(ghi_est,'r')
plt.show()



# %%
fits = best_model.predict(X_train)
plt.figure
plt.plot(y_train)
plt.plot(fits,'r')
plt.show()

plt.figure
plt.plot(ghi_true,ghi_est,'kx')
plt.show()


plt.figure
plt.plot(y_train,fits,'kx')
plt.show()

# x0=X_val[:,0]
# pred_non_up=[]
# for i in range(100):
# 	pred_non_up.append(best_model.predict(x0))

# pred_non_up=np.array(pred_non_up)
# plt.figure
# plt.plot(ghi_true)
# plt.plot(pred_non_up)
# plt.show()


