import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from timeseers.linear_trend import LinearTrend
from timeseers.fourier_seasonality import FourierSeasonality
from timeseers.regressor import Regressor
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

# select groups and order
dct_gs = {
    7226: 'muenchen_maxvorstadt', 
    7228: 'muenchen_altstadt',    
    7212: 'nuernberg_nord_fuerth',
    7231: 'regensburg',            
    7211: 'erlangen',             
    7218: 'augsburg',             
    7213: 'nuernberg_sued',       
    7204: 'wuerzburg',            
    7235: 'landshut',
    7225: 'muenchen_laim',
    7233: 'rosenheim',
    7219: 'diedorf',
    7221: 'kempten',
    7237: 'trostberg'
}
cust_gs = []#[7228,7212,7213,7231,7204,7205]
if cust_gs:
    gs = cust_gs
else:
    gs = list(dct_gs.keys())
df = df[df.g.isin(gs)].copy().reset_index(drop=True)

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

print("finished")