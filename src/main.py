import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from timeseers.linear_trend import LinearTrend
from timeseers.fourier_seasonality import FourierSeasonality
from timeseers.regressor import Regressor
import matplotlib.pyplot as plt

# load data into dataframe
# / 

# add corona data as external regressor
df_corona = pd.read_csv('../input/data/data_corona_oxford_ger.csv', index_col=False)
df_corona['date'] = pd.to_datetime(df_corona.date)
df = df.merge(df_corona[['date','gatherings_restricted_fg']], 
              left_on='t', right_on='date', how='left')
df.drop('date', axis=1, inplace=True)
df.fillna(0, inplace=True)
df.rename({'gatherings_restricted_fg': 'corona'}, axis=1, inplace=True)

# for the library, g (and regressors) must be categorical
df['g'] = df['g'].astype('category')
df['corona'] = df['corona'].astype('category')

# split to train and test data
df_train = df[df.t<SPLIT_DATE].copy()
df_test = df[df.t>=SPLIT_DATE].copy()

# split X and y, NOTE: X must include 'g' and 't' 
X_train = df_train.copy()
y_train = X_train.pop('value')

X_test = df_test.copy()
y_test = X_test.pop('value')

# build the model
trend = LinearTrend(n_changepoints=10, growth_prior_scale=5, changepoints_prior_scale=5, pool_cols='g', pool_type='partial') 
yearly_seasonality = FourierSeasonality(n = 4, shrinkage_strength = 1, period = pd.Timedelta(days=365.25), pool_cols='g', pool_type='partial')
regressor_corona = Regressor(on='corona', pool_cols='g', pool_type='partial')
model = trend + yearly_seasonality + regressor_corona

# fit the model
model.fit(X_train, y_train, **{'chains': 1,'draws':100,'tune': 100, 'cores': 1})

# predict, NOTE: Use train and test because we (must) predict in- and out-of-sample
preds = model.predict(X_train, X_test)

# plot components (calls predict internally)
model.plot_components(X_train, y_train, X_test, y_test, groups=df['g'])