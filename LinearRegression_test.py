import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.DataFrame()
ls_area = [100,101,102,103,104]
ls_price = [20000,20001,20002,20003,20004]
df['area'] = ls_area
df['price'] = ls_price

model = LinearRegression()
model.fit(df[['area']],df.price)

model.predict([[5000]])

import pickle
with open('demo.pkl', 'wb') as files:
    pickle.dump(model, files)

with open('demo.pkl' , 'rb') as f:
    lr = pickle.load(f)
lr.predict([[5000]])

