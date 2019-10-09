# %%
import scipy.io as sio

# %%
mat_contents = sio.loadmat('ImGHIrec2.mat')

# %%


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
x=mat_Im[:,:,100]
y=mat_Im[:,:,105]

# %%
x=x.reshape(x.size,1)
y=y.reshape(y.size,1)

# %%

from matplotlib import pyplot as plt


# %%
plt.figure
plt.plot(x,y,'xk')
plt.show()
plt.figure
plt.plot(x,'r')
plt.plot(y,'b')
plt.show()


# %%


# %%
