# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:25:41 2019

@author: wuersch
"""

#%matplotlib notebook

import pandas as pd
import numpy as np


from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from datetime import datetime

print(__doc__)

#%%

df=pd.read_csv('./forex/EURCHF_1 Min_Ask_2003.08.03_2019.05.13.csv',sep=';');
print(df.shape)
print(df.head(20))

df.dropna()


#ds=pd.Series.from_csv('./forex/EURCHF_1 Min_Ask_2003.08.03_2019.05.13.csv',sep=';', index_col=0)
#print(ds.shape)
#print(ds.head(20))

#%%
date_rng = pd.date_range(start='1/1/2014', end='1/08/2015', freq='H')
type(date_rng[0])


df['datetime']   = pd.to_datetime(df['Time (UTC)'])
df['Time'] = pd.to_datetime(df['Time (UTC)'])
df = df.set_index('datetime')
df.drop(['Time (UTC)'], axis=1, inplace=True)
print(df.head())

print(df.dtypes)

#%% Only open is used


ds=df.iloc[1:10000000,1]
print(type(ds))
print(ds.head(10))

ds=ds.dropna()
ds.plot()
ds['2018-1-1':'2018-12-31'].plot()

#%%

dDay=df.resample('D').mean()
dDay.plot()
dDay=dDay.dropna()

print(dDay.dtypes)
print(dDay.head())

#%%


#getting the columns out of the dataframe
X=np.float64(dDay['2017-1-1':'2018-12-31'].index.values.reshape(-1,1))/1E9/3600/24/365
y=np.float64(dDay['2017-1-1':'2018-12-31']['Open'].values.reshape(-1,1))
X=X-X[0,0]



fig, ax = plt.subplots(1, 1, constrained_layout=True)
plt.plot(X,y)
ax.set_title('subplot 1')
ax.set_xlabel('time (datetime)')
ax.set_ylabel('Forex Open')
fig.suptitle('Gaussian Process', fontsize=16)




#%%  Kernel with parameters

#Learned kernel: 0.00316**2 * RBF(length_scale=5.59e+04) + 0.0402**2 * RBF(length_scale=0.547) * Matern(length_scale=2.43e+04, nu=1) + 0.0153**2 * RationalQuadratic(alpha=0.279, length_scale=0.0552) + 0.00316**2 * RBF(length_scale=0.00954) + WhiteKernel(noise_level=1e-05)
#Log-marginal-likelihood: 2680.725


k1 = 0.00316**2 * Matern(length_scale=0.1,nu=0.5)  # long term smooth rising trend
k2 = 0.0402**2 * RBF(length_scale=0.547) \
    * Matern(length_scale=0.05, nu=0.5)          # seasonal component
# medium term irregularity
k3 = 0.0153**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)             # noise terms
kernel_gpml = k1 + k2 + k4

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, normalize_y=True)


# fit the Gaussian model

gp.fit(X, y)

#%% plotting the prediction

#%matplotlib auto

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

X_ = np.linspace(X.min(), X.max() + 0.1, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

plt.figure()
# Illustration
plt.scatter(X, y, c='k')
plt.plot(X_, y_pred)
plt.fill_between(X_[:,0], y_pred[:,0] - y_std, y_pred[:,0] + y_std, alpha=0.5, color='k')
plt.xlim(X_.min(), X_.max())
plt.xlabel("Year")
plt.ylabel(r"Forex")
plt.title(r"Prediction using a Gaussian Process")
plt.tight_layout()
plt.show()



#%%


print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

# Kernel with optimized parameters
k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
k2 = 2.0**2 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, np.inf))  # noise terms
kernel = k1 + k2 + k3 + k4



gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              normalize_y=True)

#%%

gp.fit(X, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

# Illustration
plt.scatter(X, y, c='k')
plt.plot(X_, y_pred)
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.5, color='k')
plt.xlim(X_.min(), X_.max())
plt.xlabel("Year")
plt.ylabel(r"Forex")
plt.title(r"Forex")
plt.tight_layout()
plt.show()
