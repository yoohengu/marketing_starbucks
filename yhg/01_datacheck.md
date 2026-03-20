```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 전역 시드 설정 (재현성을 위해)
np.random.seed(42)

print("="*60)
print("라이브러리 로드 완료!")
print("한글 폰트 설정 완료!")
print("="*60)
```

    ============================================================
    라이브러리 로드 완료!
    한글 폰트 설정 완료!
    ============================================================
    

---


```python
df_pf = pd.read_csv('../data/portfolio.csv')
df_pro = pd.read_csv('../data/profile.csv')
df_tran = pd.read_csv('../data/transcript.csv')
```


```python
def check(df, name="Data"):
    print(f"{name} 기본 정보")
    
    print("\n[1] 데이터 크기")
    display(df.shape)
    
    print("\n[2] 컬럼 정보")
    df.info()
    
    print("\n[3] 결측치 개수")
    display(
    df.isnull().sum()
    .sort_values(ascending=False)
    .to_frame("결측치 개수")
    )
    
    print("\n[4] 중복 데이터 개수")
    display(df.duplicated().sum())
```

---

# 1. portfolio.csv


```python
check(df_pf, 'portfolio.csv')
```

    portfolio.csv 기본 정보
    
    [1] 데이터 크기
    


    (10, 7)


    
    [2] 컬럼 정보
    <class 'pandas.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 7 columns):
     #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
     0   Unnamed: 0  10 non-null     int64
     1   reward      10 non-null     int64
     2   channels    10 non-null     str  
     3   difficulty  10 non-null     int64
     4   duration    10 non-null     int64
     5   offer_type  10 non-null     str  
     6   offer_id    10 non-null     str  
    dtypes: int64(4), str(3)
    memory usage: 692.0 bytes
    
    [3] 결측치 개수
    


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
      <th>결측치 개수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reward</th>
      <td>0</td>
    </tr>
    <tr>
      <th>channels</th>
      <td>0</td>
    </tr>
    <tr>
      <th>difficulty</th>
      <td>0</td>
    </tr>
    <tr>
      <th>duration</th>
      <td>0</td>
    </tr>
    <tr>
      <th>offer_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>offer_id</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    [4] 중복 데이터 개수
    


    np.int64(0)



```python
df_pf.describe()
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
      <th>Unnamed: 0</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.00000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.50000</td>
      <td>4.200000</td>
      <td>7.700000</td>
      <td>6.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.02765</td>
      <td>3.583915</td>
      <td>5.831905</td>
      <td>2.321398</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.25000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.50000</td>
      <td>4.000000</td>
      <td>8.500000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.75000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.00000</td>
      <td>10.000000</td>
      <td>20.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pf.head()
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
      <th>Unnamed: 0</th>
      <th>reward</th>
      <th>channels</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>offer_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10</td>
      <td>['email', 'mobile', 'social']</td>
      <td>10</td>
      <td>7</td>
      <td>bogo</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10</td>
      <td>['web', 'email', 'mobile', 'social']</td>
      <td>10</td>
      <td>5</td>
      <td>bogo</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>['web', 'email', 'mobile']</td>
      <td>0</td>
      <td>4</td>
      <td>informational</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>['web', 'email', 'mobile']</td>
      <td>5</td>
      <td>7</td>
      <td>bogo</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>['web', 'email']</td>
      <td>20</td>
      <td>10</td>
      <td>discount</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'{df_pf['reward'].value_counts(dropna=False)}\n')
print(f'{df_pf['channels'].value_counts(dropna=False)}\n')
print(f'{df_pf['difficulty'].value_counts(dropna=False)}\n')
print(f'{df_pf['duration'].value_counts(dropna=False)}\n')
print(f'{df_pf['offer_type'].value_counts(dropna=False)}')
```

    reward
    5     3
    10    2
    0     2
    2     2
    3     1
    Name: count, dtype: int64
    
    channels
    ['web', 'email', 'mobile', 'social']    4
    ['web', 'email', 'mobile']              3
    ['email', 'mobile', 'social']           2
    ['web', 'email']                        1
    Name: count, dtype: int64
    
    difficulty
    10    4
    0     2
    5     2
    20    1
    7     1
    Name: count, dtype: int64
    
    duration
    7     4
    5     2
    10    2
    4     1
    3     1
    Name: count, dtype: int64
    
    offer_type
    bogo             4
    discount         4
    informational    2
    Name: count, dtype: int64
    

---

# 2. profile.csv


```python
check(df_pro, 'profile.csv')
```

    profile.csv 기본 정보
    
    [1] 데이터 크기
    


    (17000, 6)


    
    [2] 컬럼 정보
    <class 'pandas.DataFrame'>
    RangeIndex: 17000 entries, 0 to 16999
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Unnamed: 0        17000 non-null  int64  
     1   gender            14825 non-null  str    
     2   age               17000 non-null  int64  
     3   id                17000 non-null  str    
     4   became_member_on  17000 non-null  int64  
     5   income            14825 non-null  float64
    dtypes: float64(1), int64(3), str(2)
    memory usage: 797.0 KB
    
    [3] 결측치 개수
    


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
      <th>결측치 개수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gender</th>
      <td>2175</td>
    </tr>
    <tr>
      <th>income</th>
      <td>2175</td>
    </tr>
    <tr>
      <th>Unnamed: 0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0</td>
    </tr>
    <tr>
      <th>id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>became_member_on</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    [4] 중복 데이터 개수
    


    np.int64(0)



```python
df_pro.describe()
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
      <th>Unnamed: 0</th>
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17000.000000</td>
      <td>17000.000000</td>
      <td>1.700000e+04</td>
      <td>14825.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8499.500000</td>
      <td>62.531412</td>
      <td>2.016703e+07</td>
      <td>65404.991568</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4907.621624</td>
      <td>26.738580</td>
      <td>1.167750e+04</td>
      <td>21598.299410</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>2.013073e+07</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4249.750000</td>
      <td>45.000000</td>
      <td>2.016053e+07</td>
      <td>49000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8499.500000</td>
      <td>58.000000</td>
      <td>2.017080e+07</td>
      <td>64000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12749.250000</td>
      <td>73.000000</td>
      <td>2.017123e+07</td>
      <td>80000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16999.000000</td>
      <td>118.000000</td>
      <td>2.018073e+07</td>
      <td>120000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pro.head()
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
      <th>Unnamed: 0</th>
      <th>gender</th>
      <th>age</th>
      <th>id</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
      <td>118</td>
      <td>68be06ca386d4c31939f3a4f0e3dd783</td>
      <td>20170212</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>F</td>
      <td>55</td>
      <td>0610b486422d4921ae7d2bf64640c50b</td>
      <td>20170715</td>
      <td>112000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>NaN</td>
      <td>118</td>
      <td>38fe809add3b4fcf9315a9694bb96ff5</td>
      <td>20180712</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>F</td>
      <td>75</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>20170509</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>NaN</td>
      <td>118</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>20170804</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'{df_pro['gender'].value_counts(dropna=False)}\n')
print(f'{df_pro['age'].value_counts(dropna=False).sort_index(ascending=False)}\n')
print(f'{df_pro['became_member_on'].value_counts(dropna=False)}\n')
print(f'{df_pro['income'].value_counts(dropna=False)}')
```

    gender
    M      8484
    F      6129
    NaN    2175
    O       212
    Name: count, dtype: int64
    
    age
    118    2175
    101       5
    100      12
    99        5
    98        5
           ... 
    22      131
    21      140
    20      135
    19      135
    18       70
    Name: count, Length: 85, dtype: int64
    
    became_member_on
    20171207    43
    20170819    42
    20171007    40
    20171113    39
    20180125    38
                ..
    20140109     1
    20150707     1
    20140421     1
    20140605     1
    20130922     1
    Name: count, Length: 1716, dtype: int64
    
    income
    NaN         2175
    73000.0      314
    72000.0      297
    71000.0      294
    57000.0      288
                ... 
    116000.0      46
    112000.0      45
    107000.0      45
    117000.0      32
    120000.0      13
    Name: count, Length: 92, dtype: int64
    


```python
df_pro['age_group'] = pd.cut(
    df_pro['age'],
    bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 99, 199],
    labels=['10대 이하', '20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대', '100세 이상']
)
```


```python
# df_pro.to_csv(r'C:\Users\hkhk3\Desktop\sparta\project\tp3\marketing_starbucks\data\profile_ag.csv', index=False, encoding='utf-8-sig')
```


```python
ag_cnt = df_pro['age_group'].value_counts(dropna=False)
ag_rat = df_pro['age_group'].value_counts(normalize=True, dropna=False)

display(pd.concat([ag_cnt, ag_rat], axis=1, keys=['count', 'ratio']).sort_index(ascending=False).round(2))
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
      <th>count</th>
      <th>ratio</th>
    </tr>
    <tr>
      <th>age_group</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100세 이상</th>
      <td>2192</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>90대</th>
      <td>254</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>80대</th>
      <td>831</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>70대</th>
      <td>1782</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>60대</th>
      <td>2991</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>50대</th>
      <td>3541</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>40대</th>
      <td>2309</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>30대</th>
      <td>1526</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>20대</th>
      <td>1369</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>10대 이하</th>
      <td>205</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>


---

# 3. transcript.csv


```python
check(df_tran, 'transcript.csv')
```

    transcript.csv 기본 정보
    
    [1] 데이터 크기
    


    (306534, 5)


    
    [2] 컬럼 정보
    <class 'pandas.DataFrame'>
    RangeIndex: 306534 entries, 0 to 306533
    Data columns (total 5 columns):
     #   Column      Non-Null Count   Dtype
    ---  ------      --------------   -----
     0   Unnamed: 0  306534 non-null  int64
     1   person      306534 non-null  str  
     2   event       306534 non-null  str  
     3   value       306534 non-null  str  
     4   time        306534 non-null  int64
    dtypes: int64(2), str(3)
    memory usage: 11.7 MB
    
    [3] 결측치 개수
    


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
      <th>결측치 개수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>person</th>
      <td>0</td>
    </tr>
    <tr>
      <th>event</th>
      <td>0</td>
    </tr>
    <tr>
      <th>value</th>
      <td>0</td>
    </tr>
    <tr>
      <th>time</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    [4] 중복 데이터 개수
    


    np.int64(0)



```python
df_tran.describe()
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
      <th>Unnamed: 0</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>306534.000000</td>
      <td>306534.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>153266.500000</td>
      <td>366.382940</td>
    </tr>
    <tr>
      <th>std</th>
      <td>88488.888045</td>
      <td>200.326314</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>76633.250000</td>
      <td>186.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>153266.500000</td>
      <td>408.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>229899.750000</td>
      <td>528.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>306533.000000</td>
      <td>714.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_tran.head(30)
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
      <th>Unnamed: 0</th>
      <th>person</th>
      <th>event</th>
      <th>value</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>offer received</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8ec6ce2a7e7949b1bf142def7d0e0586</td>
      <td>offer received</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>68617ca6246f4fbc85e91a2a49552598</td>
      <td>offer received</td>
      <td>{'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>389bc3fa690240e798340f5a15918d5c</td>
      <td>offer received</td>
      <td>{'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>c4863c7985cf408faee930f111475da3</td>
      <td>offer received</td>
      <td>{'offer id': '2298d6c36e964ae4a3e7e9706d1fb8c2'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>2eeac8d8feae4a8cad5a6af0499a211d</td>
      <td>offer received</td>
      <td>{'offer id': '3f207df678b143eea3cee63160fa8bed'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>aa4862eba776480b8bb9c68455b8c2e1</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>31dda685af34476cad5bc968bdb01c53</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>744d603ef08c4f33af5a61c8c7628d1c</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>3d02345581554e81b7b289ab5e288078</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>4b0da7e80e5945209a1fdddfe813dbe0</td>
      <td>offer received</td>
      <td>{'offer id': 'ae264e3637204a6fb9bb56bc8210ddfd'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>c27e0d6ab72c455a8bb66d980963de60</td>
      <td>offer received</td>
      <td>{'offer id': '3f207df678b143eea3cee63160fa8bed'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>d53717f5400c4e84affdaeda9dd926b3</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>f806632c011441378d4646567f357a21</td>
      <td>offer received</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>d058f73bf8674a26a95227db098147b1</td>
      <td>offer received</td>
      <td>{'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>65aba5c617294649aeb624da249e1ee5</td>
      <td>offer received</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>ebe7ef46ea6f4963a7dd49f501b26779</td>
      <td>offer received</td>
      <td>{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>1e9420836d554513ab90eba98552d0a9</td>
      <td>offer received</td>
      <td>{'offer id': 'ae264e3637204a6fb9bb56bc8210ddfd'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>868317b9be554cb18e50bc68484749a2</td>
      <td>offer received</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>f082d80f0aac47a99173ba8ef8fc1909</td>
      <td>offer received</td>
      <td>{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>102e9454054946fda62242d2e176fdce</td>
      <td>offer received</td>
      <td>{'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>4beeb3ed64dd4898b0edf2f6b67426d3</td>
      <td>offer received</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>9f30b375d7bd4c62a884ffe7034e09ee</td>
      <td>offer received</td>
      <td>{'offer id': '2298d6c36e964ae4a3e7e9706d1fb8c2'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>25c906289d154b66bf579693f89481c9</td>
      <td>offer received</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>6e014185620b49bd98749f728747572f</td>
      <td>offer received</td>
      <td>{'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>02c083884c7d45b39cc68e1314fec56c</td>
      <td>offer received</td>
      <td>{'offer id': 'ae264e3637204a6fb9bb56bc8210ddfd'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>c0d210398dee4a0895b24444a5fcd1d2</td>
      <td>offer received</td>
      <td>{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>8be4463721e14d7fa600686bf8c8b2ed</td>
      <td>offer received</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'{df_tran['event'].value_counts(dropna=False)}\n')
print(f'{df_tran['value'].value_counts(dropna=False)}\n')
print(f'{df_tran['time'].value_counts(dropna=False)}\n')
```

    event
    transaction        138953
    offer received      76277
    offer viewed        57725
    offer completed     33579
    Name: count, dtype: int64
    
    value
    {'offer id': '2298d6c36e964ae4a3e7e9706d1fb8c2'}    14983
    {'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}    14924
    {'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}    14891
    {'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}    14835
    {'offer id': 'ae264e3637204a6fb9bb56bc8210ddfd'}    14374
                                                        ...  
    {'amount': 290.93}                                      1
    {'amount': 43.91}                                       1
    {'amount': 685.07}                                      1
    {'amount': 405.04}                                      1
    {'amount': 476.33}                                      1
    Name: count, Length: 5121, dtype: int64
    
    time
    408    17030
    576    17015
    504    16822
    336    16302
    168    16150
           ...  
    318      940
    330      938
    156      914
    162      910
    150      894
    Name: count, Length: 120, dtype: int64
    
    

---


```python
!jupyter nbconvert --to markdown "01_datacheck.ipynb"
```

    [NbConvertApp] Converting notebook 01_datacheck.ipynb to markdown
    [NbConvertApp] Writing 20707 bytes to 01_datacheck.md
    
