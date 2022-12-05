import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from timeseers.linear_trend import LinearTrend
from timeseers.fourier_seasonality import FourierSeasonality
from timeseers.constant import Constant
import matplotlib.pyplot as plt


# constants
SPLIT_DATE = '2022-07-01'

# load data
df = pd.read_csv('../input/data/dovato_sales.csv', index_col=False).assign(
    t=lambda d: pd.to_datetime(d['date_month']),
    value=lambda d: d['days_of_treatment']
)
df.rename({'gross_segment':'g'}, axis=1, inplace=True)
df = df[['t','g','value']]

# select groups
df = df[df.g.isin([7226,7228])].copy().reset_index(drop=True)

# for the library, g must be categorical
df['g'] = df['g'].astype('category')

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
yearly_seasonality = FourierSeasonality(n = 5, shrinkage_strength = 1, period = pd.Timedelta(days=365.25), pool_cols='g', pool_type='partial')
model = trend + yearly_seasonality

# fit the model
model.fit(X_train, y_train, **{'chains': 1,'draws':50,'tune': 50, 'cores': 1})

# predict, NOTE: Use train and test because we (must) predict in- and out-of-sample
preds = model.predict(X_train, X_test)

# plot components
y = pd.concat([y_train, y_test])
X = pd.concat([X_train, X_test])
model.plot_components(X, y, groups=df['g'])

print('fin')