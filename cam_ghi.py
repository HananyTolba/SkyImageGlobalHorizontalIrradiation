# %%
import scipy.io as sio

# %%
mat_contents = sio.loadmat('C:\\Users\\hanany\\Documents\\cam_DeepLearning\\ImGHIrec2.mat')

# %%


# %%


# %%
DateTimeCom=sio.loadmat('C:\\Users\\hanany\\Documents\\cam_DeepLearning\\DateTimeCom.mat')
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

# %%

from matplotlib import pyplot as plt


# %%
plt.plot(x)
plt.plot(y)
plt.show()

# %%


# %%
