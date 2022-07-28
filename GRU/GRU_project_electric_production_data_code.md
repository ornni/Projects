```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_check=pd.read_csv('C:\\Users\\user\\Desktop\\data\\time_series_dataset_archive\\Electric_Production.csv')
data_check.head()

data=pd.read_csv('C:\\Users\\user\\Desktop\\data\\time_series_dataset_archive\\Electric_Production.csv', parse_dates=['DATE'], index_col='DATE')
data.head()

df=pd.DataFrame(data)
df=data.rename(columns={'DATE':'date', 'IPG2211A2N': 'EP'})
df.shape
df.head()

plt.figure(figsize=(20, 10))
plt.plot(df)
plt.show()
```


    
![2-1](https://github.com/ornni/DL_algorithm/blob/main/GRU/image/GRU_project_2-1.png?raw=true)
    



```python
# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

df['EP']=scaler.fit_transform(df)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EP</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1985-01-01</th>
      <td>0.232017</td>
    </tr>
    <tr>
      <th>1985-02-01</th>
      <td>0.207274</td>
    </tr>
    <tr>
      <th>1985-03-01</th>
      <td>0.096304</td>
    </tr>
    <tr>
      <th>1985-04-01</th>
      <td>0.029104</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-09-01</th>
      <td>0.584431</td>
    </tr>
    <tr>
      <th>2017-10-01</th>
      <td>0.516922</td>
    </tr>
    <tr>
      <th>2017-11-01</th>
      <td>0.567161</td>
    </tr>
    <tr>
      <th>2017-12-01</th>
      <td>0.801813</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>397 rows × 1 columns</p>
</div>




```python
# 다음 달의 값을 정답 레이블로 추가
df['EP_t+1']=df['EP'].shift(-1)
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EP</th>
      <th>EP_t+1</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-09-01</th>
      <td>0.584431</td>
      <td>0.516922</td>
    </tr>
    <tr>
      <th>2017-10-01</th>
      <td>0.516922</td>
      <td>0.567161</td>
    </tr>
    <tr>
      <th>2017-11-01</th>
      <td>0.567161</td>
      <td>0.801813</td>
    </tr>
    <tr>
      <th>2017-12-01</th>
      <td>0.801813</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 이전 3개의 시점을 사용하여 t+1의 값을 예측할 예정
for i in range(4):
    df[f'EP_t-{i}']=df['EP'].shift(+i)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EP</th>
      <th>EP_t+1</th>
      <th>EP_t-0</th>
      <th>EP_t-1</th>
      <th>EP_t-2</th>
      <th>EP_t-3</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1985-01-01</th>
      <td>0.232017</td>
      <td>0.207274</td>
      <td>0.232017</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-02-01</th>
      <td>0.207274</td>
      <td>0.096304</td>
      <td>0.207274</td>
      <td>0.232017</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-03-01</th>
      <td>0.096304</td>
      <td>0.029104</td>
      <td>0.096304</td>
      <td>0.207274</td>
      <td>0.232017</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1985-04-01</th>
      <td>0.029104</td>
      <td>0.000000</td>
      <td>0.029104</td>
      <td>0.096304</td>
      <td>0.207274</td>
      <td>0.232017</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>0.000000</td>
      <td>0.037459</td>
      <td>0.000000</td>
      <td>0.029104</td>
      <td>0.096304</td>
      <td>0.207274</td>
    </tr>
  </tbody>
</table>
</div>




```python
# NaN 데이터가 들어간 경우 삭제
df.dropna(inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EP</th>
      <th>EP_t+1</th>
      <th>EP_t-0</th>
      <th>EP_t-1</th>
      <th>EP_t-2</th>
      <th>EP_t-3</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1985-04-01</th>
      <td>0.029104</td>
      <td>0.000000</td>
      <td>0.029104</td>
      <td>0.096304</td>
      <td>0.207274</td>
      <td>0.232017</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>0.000000</td>
      <td>0.037459</td>
      <td>0.000000</td>
      <td>0.029104</td>
      <td>0.096304</td>
      <td>0.207274</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>0.037459</td>
      <td>0.098598</td>
      <td>0.037459</td>
      <td>0.000000</td>
      <td>0.029104</td>
      <td>0.096304</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>0.098598</td>
      <td>0.107078</td>
      <td>0.098598</td>
      <td>0.037459</td>
      <td>0.000000</td>
      <td>0.029104</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>0.107078</td>
      <td>0.071123</td>
      <td>0.107078</td>
      <td>0.098598</td>
      <td>0.037459</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (393, 6)




```python
# 데이터 분할 위치 확인(날짜 확인)
print(df.iloc[196, :], df.iloc[196+99, :])
```

    EP        0.597091
    EP_t+1    0.444946
    EP_t-0    0.597091
    EP_t-1    0.550129
    EP_t-2    0.473486
    EP_t-3    0.367546
    Name: 2001-08-01 00:00:00, dtype: float64 EP        0.459489
    EP_t+1    0.747328
    EP_t-0    0.459489
    EP_t-1    0.458668
    EP_t-2    0.534045
    EP_t-3    0.680709
    Name: 2009-11-01 00:00:00, dtype: float64
    


```python
# 데이터 분할
valid_start='2001-08-01 00:00:00'
test_start='2009-11-01 00:00:00'

train_data=df[df.index<valid_start]
valid_data=df[(df.index>=valid_start) & (df.index<test_start)] 
test_data=df[df.index>=test_start]

print(train_data.shape, valid_data.shape, test_data.shape)
```

    (196, 6) (99, 6) (98, 6)
    


```python
plt.figure(figsize=(20, 10))
plt.plot(train_data['EP'])
plt.plot(valid_data['EP'])
plt.plot(test_data['EP'])
plt.show()
```


    
![3-1](https://github.com/ornni/DL_algorithm/blob/main/GRU/image/GRU_project_3-1.png?raw=true)
    



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EP</th>
      <th>EP_t+1</th>
      <th>EP_t-0</th>
      <th>EP_t-1</th>
      <th>EP_t-2</th>
      <th>EP_t-3</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1985-04-01</th>
      <td>0.029104</td>
      <td>0.000000</td>
      <td>0.029104</td>
      <td>0.096304</td>
      <td>0.207274</td>
      <td>0.232017</td>
    </tr>
    <tr>
      <th>1985-05-01</th>
      <td>0.000000</td>
      <td>0.037459</td>
      <td>0.000000</td>
      <td>0.029104</td>
      <td>0.096304</td>
      <td>0.207274</td>
    </tr>
    <tr>
      <th>1985-06-01</th>
      <td>0.037459</td>
      <td>0.098598</td>
      <td>0.037459</td>
      <td>0.000000</td>
      <td>0.029104</td>
      <td>0.096304</td>
    </tr>
    <tr>
      <th>1985-07-01</th>
      <td>0.098598</td>
      <td>0.107078</td>
      <td>0.098598</td>
      <td>0.037459</td>
      <td>0.000000</td>
      <td>0.029104</td>
    </tr>
    <tr>
      <th>1985-08-01</th>
      <td>0.107078</td>
      <td>0.071123</td>
      <td>0.107078</td>
      <td>0.098598</td>
      <td>0.037459</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# x 데이터와 y 데이터 분할하기
x_train=train_data.loc[:, 'EP_t-0':'EP_t-3']
y_train=train_data['EP_t+1']

x_valid=valid_data.loc[:, 'EP_t-0':'EP_t-3']
y_valid=valid_data['EP_t+1']

x_test=test_data.loc[:, 'EP_t-0':'EP_t-3']
y_test=test_data['EP_t+1']

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)
```

    (196, 4) (196,)
    (99, 4) (99,)
    (98, 4) (98,)
    


```python
# feature값 추가
x_train=x_train.reshape(196, 4, 1)
x_valid=x_valid.reshape(99, 4, 1)
x_test=x_test.reshape(98, 4, 1)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    c:\Users\user\Desktop\python\time_series_data.py in <module>
          <a href='file:///c%3A/Users/user/Desktop/python/time_series_data.py?line=88'>89</a> # %%
          <a href='file:///c%3A/Users/user/Desktop/python/time_series_data.py?line=89'>90</a> # feature값 추가
    ----> <a href='file:///c%3A/Users/user/Desktop/python/time_series_data.py?line=90'>91</a> x_train=x_train.reshape(196, 4, 1)
          <a href='file:///c%3A/Users/user/Desktop/python/time_series_data.py?line=91'>92</a> x_valid=x_valid.reshape(99, 4, 1)
          <a href='file:///c%3A/Users/user/Desktop/python/time_series_data.py?line=92'>93</a> x_test=x_test.reshape(98, 4, 1)
    

    c:\Users\user\anaconda3\lib\site-packages\pandas\core\generic.py in __getattr__(self, name)
       5485         ):
       5486             return self[name]
    -> 5487         return object.__getattribute__(self, name)
       5488 
       5489     def __setattr__(self, name: str, value) -> None:
    

    AttributeError: 'DataFrame' object has no attribute 'reshape'



```python
# modeling
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

model=Sequential()
model.add(LSTM(32, input_shape=(4, 1)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam')

history=model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=100)
```

    Epoch 1/100
    7/7 [==============================] - 2s 91ms/step - loss: 0.0839 - val_loss: 0.1946
    Epoch 2/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0507 - val_loss: 0.1189
    Epoch 3/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0277 - val_loss: 0.0645
    Epoch 4/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0158 - val_loss: 0.0338
    Epoch 5/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0134 - val_loss: 0.0230
    Epoch 6/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0139 - val_loss: 0.0219
    Epoch 7/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0136 - val_loss: 0.0248
    Epoch 8/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0128 - val_loss: 0.0303
    Epoch 9/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0130 - val_loss: 0.0336
    Epoch 10/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0129 - val_loss: 0.0307
    Epoch 11/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0127 - val_loss: 0.0265
    Epoch 12/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0126 - val_loss: 0.0241
    Epoch 13/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0124 - val_loss: 0.0256
    Epoch 14/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0123 - val_loss: 0.0276
    Epoch 15/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0125 - val_loss: 0.0289
    Epoch 16/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0124 - val_loss: 0.0274
    Epoch 17/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0122 - val_loss: 0.0248
    Epoch 18/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0120 - val_loss: 0.0226
    Epoch 19/100
    7/7 [==============================] - 0s 6ms/step - loss: 0.0121 - val_loss: 0.0215
    Epoch 20/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0120 - val_loss: 0.0226
    Epoch 21/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0119 - val_loss: 0.0234
    Epoch 22/100
    7/7 [==============================] - 0s 6ms/step - loss: 0.0119 - val_loss: 0.0235
    Epoch 23/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0118 - val_loss: 0.0220
    Epoch 24/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0117 - val_loss: 0.0218
    Epoch 25/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0117 - val_loss: 0.0202
    Epoch 26/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0120 - val_loss: 0.0193
    Epoch 27/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0118 - val_loss: 0.0217
    Epoch 28/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0118 - val_loss: 0.0236
    Epoch 29/100
    7/7 [==============================] - 0s 6ms/step - loss: 0.0116 - val_loss: 0.0207
    Epoch 30/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0116 - val_loss: 0.0201
    Epoch 31/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0114 - val_loss: 0.0215
    Epoch 32/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0115 - val_loss: 0.0217
    Epoch 33/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0114 - val_loss: 0.0201
    Epoch 34/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0114 - val_loss: 0.0204
    Epoch 35/100
    7/7 [==============================] - 0s 10ms/step - loss: 0.0114 - val_loss: 0.0225
    Epoch 36/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0115 - val_loss: 0.0224
    Epoch 37/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0114 - val_loss: 0.0201
    Epoch 38/100
    7/7 [==============================] - 0s 9ms/step - loss: 0.0113 - val_loss: 0.0199
    Epoch 39/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0112 - val_loss: 0.0210
    Epoch 40/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0112 - val_loss: 0.0210
    Epoch 41/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0112 - val_loss: 0.0210
    Epoch 42/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0112 - val_loss: 0.0202
    Epoch 43/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0111 - val_loss: 0.0208
    Epoch 44/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0112 - val_loss: 0.0222
    Epoch 45/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0112 - val_loss: 0.0214
    Epoch 46/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0112 - val_loss: 0.0191
    Epoch 47/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0111 - val_loss: 0.0194
    Epoch 48/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0110 - val_loss: 0.0197
    Epoch 49/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0110 - val_loss: 0.0214
    Epoch 50/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0112 - val_loss: 0.0229
    Epoch 51/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0112 - val_loss: 0.0203
    Epoch 52/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0109 - val_loss: 0.0204
    Epoch 53/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0109 - val_loss: 0.0205
    Epoch 54/100
    7/7 [==============================] - 0s 6ms/step - loss: 0.0110 - val_loss: 0.0213
    Epoch 55/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0109 - val_loss: 0.0183
    Epoch 56/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0108 - val_loss: 0.0188
    Epoch 57/100
    7/7 [==============================] - 0s 6ms/step - loss: 0.0107 - val_loss: 0.0193
    Epoch 58/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0107 - val_loss: 0.0213
    Epoch 59/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0108 - val_loss: 0.0189
    Epoch 60/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0106 - val_loss: 0.0186
    Epoch 61/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0106 - val_loss: 0.0185
    Epoch 62/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0106 - val_loss: 0.0190
    Epoch 63/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0105 - val_loss: 0.0182
    Epoch 64/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0105 - val_loss: 0.0195
    Epoch 65/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0105 - val_loss: 0.0176
    Epoch 66/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0105 - val_loss: 0.0182
    Epoch 67/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0104 - val_loss: 0.0179
    Epoch 68/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0103 - val_loss: 0.0197
    Epoch 69/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0103 - val_loss: 0.0189
    Epoch 70/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0102 - val_loss: 0.0176
    Epoch 71/100
    7/7 [==============================] - 0s 12ms/step - loss: 0.0103 - val_loss: 0.0181
    Epoch 72/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0101 - val_loss: 0.0197
    Epoch 73/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0101 - val_loss: 0.0184
    Epoch 74/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0101 - val_loss: 0.0174
    Epoch 75/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0101 - val_loss: 0.0182
    Epoch 76/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0100 - val_loss: 0.0189
    Epoch 77/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0099 - val_loss: 0.0176
    Epoch 78/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0100 - val_loss: 0.0180
    Epoch 79/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0101 - val_loss: 0.0165
    Epoch 80/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0101 - val_loss: 0.0195
    Epoch 81/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0099 - val_loss: 0.0168
    Epoch 82/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0098 - val_loss: 0.0237
    Epoch 83/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0103 - val_loss: 0.0204
    Epoch 84/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0096 - val_loss: 0.0154
    Epoch 85/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0109 - val_loss: 0.0152
    Epoch 86/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0098 - val_loss: 0.0215
    Epoch 87/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0096 - val_loss: 0.0164
    Epoch 88/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0096 - val_loss: 0.0168
    Epoch 89/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0095 - val_loss: 0.0206
    Epoch 90/100
    7/7 [==============================] - 0s 6ms/step - loss: 0.0094 - val_loss: 0.0164
    Epoch 91/100
    7/7 [==============================] - 0s 6ms/step - loss: 0.0095 - val_loss: 0.0167
    Epoch 92/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0094 - val_loss: 0.0201
    Epoch 93/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0092 - val_loss: 0.0160
    Epoch 94/100
    7/7 [==============================] - 0s 6ms/step - loss: 0.0092 - val_loss: 0.0183
    Epoch 95/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0091 - val_loss: 0.0171
    Epoch 96/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0091 - val_loss: 0.0162
    Epoch 97/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0091 - val_loss: 0.0164
    Epoch 98/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0090 - val_loss: 0.0188
    Epoch 99/100
    7/7 [==============================] - 0s 7ms/step - loss: 0.0089 - val_loss: 0.0167
    Epoch 100/100
    7/7 [==============================] - 0s 8ms/step - loss: 0.0088 - val_loss: 0.0172
    


```python
plt.figure(figsize=(20, 10))
plt.plot(history.history['loss'], label=['loss'])
plt.plot(history.history['val_loss'], label=['val_loss'])
plt.legend()
plt.show()
```


    
![4-1](https://github.com/ornni/DL_algorithm/blob/main/GRU/image/GRU_project_4-1.png?raw=true)
    



```python
result=test_data['EP_t+1']
result=pd.DataFrame(result)
result['y_pred']=model.predict(x_test)
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EP_t+1</th>
      <th>y_pred</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-11-01</th>
      <td>0.747328</td>
      <td>0.432570</td>
    </tr>
    <tr>
      <th>2009-12-01</th>
      <td>0.859789</td>
      <td>0.559542</td>
    </tr>
    <tr>
      <th>2010-01-01</th>
      <td>0.745284</td>
      <td>0.635479</td>
    </tr>
    <tr>
      <th>2010-02-01</th>
      <td>0.579731</td>
      <td>0.587535</td>
    </tr>
    <tr>
      <th>2010-03-01</th>
      <td>0.418208</td>
      <td>0.506472</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-08-01</th>
      <td>0.584431</td>
      <td>0.579421</td>
    </tr>
    <tr>
      <th>2017-09-01</th>
      <td>0.516922</td>
      <td>0.514544</td>
    </tr>
    <tr>
      <th>2017-10-01</th>
      <td>0.567161</td>
      <td>0.470648</td>
    </tr>
    <tr>
      <th>2017-11-01</th>
      <td>0.801813</td>
      <td>0.486524</td>
    </tr>
    <tr>
      <th>2017-12-01</th>
      <td>1.000000</td>
      <td>0.591334</td>
    </tr>
  </tbody>
</table>
<p>98 rows × 2 columns</p>
</div>




```python
inverse_result=scaler.inverse_transform(result)
inverse_result
print(type(inverse_result))
```

    <class 'numpy.ndarray'>
    


```python
plt.figure(figsize=(20, 10))
plt.plot(inverse_result[:, 0], label='real')
plt.plot(inverse_result[:, 1], label='predict')
plt.legend()
plt.show()
```


    
![5-1](https://github.com/ornni/DL_algorithm/blob/main/GRU/image/GRU_project_5-1.png?raw=true)
    



```python
# 평가지표 적용
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

mse=mean_squared_error(test_data['EP_t+1'], model.predict(x_test))
mae=mean_absolute_error(test_data['EP_t+1'], model.predict(x_test))

print('MSE: ', mse)
print('MAE: ', mae)
```

    MSE:  0.028442397983576485
    MAE:  0.13263246508405258
    
