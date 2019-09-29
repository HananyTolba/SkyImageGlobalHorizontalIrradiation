# %%
import scipy.io as sio

# %%
import os

# %%
mat_contents = sio.loadmat('/Users/hanany/Downloads/cam_DeepLearning/ImGHIrec2.mat')

# %%
path=os.getcwd()

# %%


# %%
DateTimeCom=sio.loadmat('/Users/hanany/Downloads/cam_DeepLearning/DateTimeCom.mat')
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
x=mat_Im[:,:,0]
y=mat_Im[:,:,5]

# %%
x=x.reshape(28*26,1)
y=y.reshape(28*26,1)


28*26

# %%

from matplotlib import pyplot as plt


# %%
plt.plot(x)
plt.plot(y)
plt.show()

# %%

mat_Im.shape[0]*mat_Im.shape[1]

# %%
dataset_orig=mat_Im.reshape(mat_Im.shape[0]*mat_Im.shape[1],mat_Im.shape[2])

dataset_orig=dataset_orig[:,:13000]


#dataset_orig=dataset_orig[:,::4]




from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset_orig)

#df_test['scaled_GHI'] = scaler.fit_transform(np.array(df_test['GHI']).reshape(-1, 1))

train_size=int(dataset_orig.shape[1]*.7)

data_train=dataset[:,:train_size]
data_test=dataset[:,train_size:]
data_train=data_train.T
data_test=data_test.T


# %%
data_train.shape,data_test.shape,dataset_orig.shape

#train_size

# %%


# %%
dimInput=1
X_train, y_train = data_train[:-1,:],data_train[dimInput:,:]
X_val, y_val = data_test[:-1,:],data_test[dimInput:,:]

# %%
X_val.shape,y_val.shape,X_train.shape,y_train.shape


# %%
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

# %%
#definition des entrées. None indicates the number of instances
input_layer = Input(shape=(728,), dtype='float32')

# %%
#Les couches Denses avec une fonction d'activation relu
dense1 = Dense(32, activation='relu')(input_layer)
dropout_layer = Dropout(0.1)(dense1)
dense2 = Dense(128, activation='relu')(dense1)

dense3 = Dense(32, activation='relu')(dense2)
dropout_layer = Dropout(0.1)(dense3)

# dense4 = Dense(8, activation='relu')(dense3)
# dropout_layer = Dropout(0.1)(dense4)

# %%
output_layer = Dense(728, activation='relu')(dropout_layer)


# %%
ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_squared_error', optimizer='adam')

ts_model.summary()

# %%
save_weights_at = os.path.join(path, 'Cam_MLP_poids.{epoch:02d}-{val_loss:.4f}.hdf5')
print(save_weights_at)

# %%
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)


# %%
ts_model.fit(x=X_train, y=y_train, batch_size=100, epochs=10,verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
           shuffle=True)


# %%


# %%


# %%
files = sorted(os.listdir(path), key=os.path.getctime)

oldest = files[0]
newest = files[-1]
print(newest)

# %%
os.path.join(path,newest)

# %%
best_model = load_model(os.path.join(path,newest))
preds = best_model.predict(X_val)
pred_PRES = preds
pred_PRES = np.squeeze(pred_PRES)


from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_val, pred_PRES)

#print('R-squared for the validation set:'.format(r2))
print("R-squared en test  est {} ".format(r2))

# calculate root mean squared error
testScore = np.sqrt(mean_squared_error(y_val, pred_PRES))
print("L'erreur en test  est {} (nRMSE)".format(testScore))



# %%

IM=y_val[59,:]
IM=IM.reshape(28, 26)

plt.imshow(IM)


# %%
IM=pred_PRES[59,:]
IM=IM.reshape(28, 26)

plt.imshow(IM)



# %%
plt.plot(y_val[100,:])


plt.plot(pred_PRES[100,:])
plt.show()

# %%
# %%
plt.show()

# %%
import glob
path = os.getcwd()

file_list=glob.glob(path + "**/*.hdf5", recursive=True)
#print(file_list)
for f in file_list :
    os.remove(f)
#cprint("Tout les fichers hdf5 ont été supprimés !".center(50).upper(),'red')


# %%


# %%


# %%
