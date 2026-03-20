```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display
import warnings

import ast

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
    


```python
df_pf = pd.read_csv('../data/portfolio.csv')
df_pro = pd.read_csv('../data/profile_ag.csv')
df_tran = pd.read_csv('../data/transcript.csv')
```

---

# profile.csv

## 1. 결측치 확인

- `gender`, `income`, `age` 컬럼의 결측치 수가 2175로 동일하여,<br>해당 결측치가 동일한 행에서 발생한 것인지 확인 


```python
# 1. 조건 정의
cond_gender = df_pro['gender'].isnull()
cond_income = df_pro['income'].isnull()
cond_age = df_pro['age'] == 118

# 2. id 집합 생성
set_gender = set(df_pro.loc[cond_gender, 'id'])
set_income = set(df_pro.loc[cond_income, 'id'])
set_age = set(df_pro.loc[cond_age, 'id'])

# 3. 각각 개수 확인
print("gender 결측 수:", len(set_gender))
print("income 결측 수:", len(set_income))
print("age == 118 수:", len(set_age))
print("-" * 40)

# 4. 집합 비교 (완전 동일 여부)
print("gender == income:", set_gender == set_income)
print("gender == age:", set_gender == set_age)
print("income == age:", set_income == set_age)
print("-" * 40)

# 5. 교집합 확인
print("gender ∩ income:", len(set_gender & set_income))
print("gender ∩ age:", len(set_gender & set_age))
print("income ∩ age:", len(set_income & set_age))
print("-" * 40)

# 6. 차이 확인 (어디가 다른지)
print("gender - income:", len(set_gender - set_income))
print("income - gender:", len(set_income - set_gender))
print("gender - age:", len(set_gender - set_age))
print("age - gender:", len(set_age - set_gender))
print("-" * 40)

# 7. 완전 동일 패턴 여부 (3개 다 동일)
all_equal = (set_gender == set_income == set_age)
print("세 조건 모두 동일한 id 집합인지:", all_equal)
```

    gender 결측 수: 2175
    income 결측 수: 2175
    age == 118 수: 2175
    ----------------------------------------
    gender == income: True
    gender == age: True
    income == age: True
    ----------------------------------------
    gender ∩ income: 2175
    gender ∩ age: 2175
    income ∩ age: 2175
    ----------------------------------------
    gender - income: 0
    income - gender: 0
    gender - age: 0
    age - gender: 0
    ----------------------------------------
    세 조건 모두 동일한 id 집합인지: True
    

> 같은 행에서 발생함을 확인

## 2. gender 컬럼의 other

- `gender` 컬럼에서 `other` 값을 어떻게 처리할까


```python
df_pro['gender'].value_counts(dropna=False, normalize=True).round(3)
```




    gender
    M      0.499
    F      0.361
    NaN    0.128
    O      0.012
    Name: proportion, dtype: float64



> `other`의 비율은 약 1%로 버려도 괜찮을 것 같음
<br> NaN 값을 어떻게 처리할지가 더 문제ㅠ

---

# transcript.csv


```python
df_tran1 = df_tran.sort_values(['person', 'time'])
df_tran1.head(30)
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
      <th>55972</th>
      <td>55972</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>{'offer id': '5a8bc65990b245e5a138643cd4eb9837'}</td>
      <td>168</td>
    </tr>
    <tr>
      <th>77705</th>
      <td>77705</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>{'offer id': '5a8bc65990b245e5a138643cd4eb9837'}</td>
      <td>192</td>
    </tr>
    <tr>
      <th>89291</th>
      <td>89291</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>{'amount': 22.16}</td>
      <td>228</td>
    </tr>
    <tr>
      <th>113605</th>
      <td>113605</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>{'offer id': '3f207df678b143eea3cee63160fa8bed'}</td>
      <td>336</td>
    </tr>
    <tr>
      <th>139992</th>
      <td>139992</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>{'offer id': '3f207df678b143eea3cee63160fa8bed'}</td>
      <td>372</td>
    </tr>
    <tr>
      <th>153401</th>
      <td>153401</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>{'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}</td>
      <td>408</td>
    </tr>
    <tr>
      <th>168412</th>
      <td>168412</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>{'amount': 8.57}</td>
      <td>414</td>
    </tr>
    <tr>
      <th>168413</th>
      <td>168413</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer completed</td>
      <td>{'offer_id': 'f19421c1d4aa40978ebb69ca19b0e20d...</td>
      <td>414</td>
    </tr>
    <tr>
      <th>187554</th>
      <td>187554</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>{'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}</td>
      <td>456</td>
    </tr>
    <tr>
      <th>204340</th>
      <td>204340</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
      <td>504</td>
    </tr>
    <tr>
      <th>228422</th>
      <td>228422</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>{'amount': 14.11}</td>
      <td>528</td>
    </tr>
    <tr>
      <th>228423</th>
      <td>228423</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer completed</td>
      <td>{'offer_id': 'fafdcd668e3743c1bb461111dcafc2a4...</td>
      <td>528</td>
    </tr>
    <tr>
      <th>233413</th>
      <td>233413</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>{'offer id': 'fafdcd668e3743c1bb461111dcafc2a4'}</td>
      <td>540</td>
    </tr>
    <tr>
      <th>237784</th>
      <td>237784</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>{'amount': 13.56}</td>
      <td>552</td>
    </tr>
    <tr>
      <th>247879</th>
      <td>247879</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>{'offer id': '2906b810c7d4411798c6938adc9daaa5'}</td>
      <td>576</td>
    </tr>
    <tr>
      <th>258883</th>
      <td>258883</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>{'amount': 10.27}</td>
      <td>576</td>
    </tr>
    <tr>
      <th>258884</th>
      <td>258884</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer completed</td>
      <td>{'offer_id': '2906b810c7d4411798c6938adc9daaa5...</td>
      <td>576</td>
    </tr>
    <tr>
      <th>293497</th>
      <td>293497</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>{'amount': 12.36}</td>
      <td>660</td>
    </tr>
    <tr>
      <th>300930</th>
      <td>300930</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>{'amount': 28.16}</td>
      <td>690</td>
    </tr>
    <tr>
      <th>302205</th>
      <td>302205</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>{'amount': 18.41}</td>
      <td>696</td>
    </tr>
    <tr>
      <th>56475</th>
      <td>56475</td>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>offer received</td>
      <td>{'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}</td>
      <td>168</td>
    </tr>
    <tr>
      <th>85769</th>
      <td>85769</td>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>offer viewed</td>
      <td>{'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}</td>
      <td>216</td>
    </tr>
    <tr>
      <th>104088</th>
      <td>104088</td>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>transaction</td>
      <td>{'amount': 0.7000000000000001}</td>
      <td>294</td>
    </tr>
    <tr>
      <th>187632</th>
      <td>187632</td>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>transaction</td>
      <td>{'amount': 0.2}</td>
      <td>456</td>
    </tr>
    <tr>
      <th>193680</th>
      <td>193680</td>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>transaction</td>
      <td>{'amount': 3.19}</td>
      <td>474</td>
    </tr>
    <tr>
      <th>248359</th>
      <td>248359</td>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>offer received</td>
      <td>{'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}</td>
      <td>576</td>
    </tr>
    <tr>
      <th>284472</th>
      <td>284472</td>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>offer viewed</td>
      <td>{'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}</td>
      <td>630</td>
    </tr>
    <tr>
      <th>3066</th>
      <td>3066</td>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>offer received</td>
      <td>{'offer id': '3f207df678b143eea3cee63160fa8bed'}</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16179</th>
      <td>16179</td>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>offer viewed</td>
      <td>{'offer id': '3f207df678b143eea3cee63160fa8bed'}</td>
      <td>6</td>
    </tr>
    <tr>
      <th>47805</th>
      <td>47805</td>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>transaction</td>
      <td>{'amount': 13.49}</td>
      <td>132</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_tran[(df_tran['time'] == 0) & (df_tran['event'] == 'offer completed')]
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
      <th>12658</th>
      <td>12658</td>
      <td>9fa9ae8f57894cc9a3b8a9bbe0fc1b2f</td>
      <td>offer completed</td>
      <td>{'offer_id': '2906b810c7d4411798c6938adc9daaa5...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12672</th>
      <td>12672</td>
      <td>fe97aa22dd3e48c8b143116a8403dd52</td>
      <td>offer completed</td>
      <td>{'offer_id': 'fafdcd668e3743c1bb461111dcafc2a4...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12679</th>
      <td>12679</td>
      <td>629fc02d56414d91bca360decdfa9288</td>
      <td>offer completed</td>
      <td>{'offer_id': '9b98b8c7a33c4b65b9aebfe6a799e6d9...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12692</th>
      <td>12692</td>
      <td>676506bad68e4161b9bbaffeb039626b</td>
      <td>offer completed</td>
      <td>{'offer_id': 'ae264e3637204a6fb9bb56bc8210ddfd...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12697</th>
      <td>12697</td>
      <td>8f7dd3b2afe14c078eb4f6e6fe4ba97d</td>
      <td>offer completed</td>
      <td>{'offer_id': '4d5c57ea9a6940dd891ad53e9dbe8da0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15521</th>
      <td>15521</td>
      <td>2c107d718e614961ae18c8ef2af37b03</td>
      <td>offer completed</td>
      <td>{'offer_id': '4d5c57ea9a6940dd891ad53e9dbe8da0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15532</th>
      <td>15532</td>
      <td>0b64be3b241c4407a5c9a71781173829</td>
      <td>offer completed</td>
      <td>{'offer_id': 'ae264e3637204a6fb9bb56bc8210ddfd...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15536</th>
      <td>15536</td>
      <td>1593d617fac246ef8e50dbb0ffd77f5f</td>
      <td>offer completed</td>
      <td>{'offer_id': '2906b810c7d4411798c6938adc9daaa5...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15541</th>
      <td>15541</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer completed</td>
      <td>{'offer_id': '4d5c57ea9a6940dd891ad53e9dbe8da0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15550</th>
      <td>15550</td>
      <td>d5d487cfdc5b456b84b7881ad5d3640b</td>
      <td>offer completed</td>
      <td>{'offer_id': 'f19421c1d4aa40978ebb69ca19b0e20d...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>206 rows × 5 columns</p>
</div>



---


```python
# value 문자열 -> dict 변환
df_tran['value'] = df_tran['value'].apply(ast.literal_eval)

# offer_id 추출 ('offer id' 또는 'offer_id' 둘 다 대응)
df_tran['offer_id'] = df_tran['value'].apply(
    lambda x: x.get('offer id') if isinstance(x, dict) and 'offer id' in x
    else (x.get('offer_id') if isinstance(x, dict) and 'offer_id' in x
    else (x.get('amount') if isinstance(x, dict) and 'amount' in x else None))
)

df_tran = df_tran.drop(columns='value')
# df_tran['offer_id'] = df_tran['value'].apply(
#     lambda x: x.get('offer id') if isinstance(x, dict) else None
# )
```


```python
df_tran1 = df_tran.sort_values(['person', 'time'])
df_tran1.head(3)
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
      <th>time</th>
      <th>offer_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55972</th>
      <td>55972</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>168</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
    </tr>
    <tr>
      <th>77705</th>
      <td>77705</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>192</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
    </tr>
    <tr>
      <th>89291</th>
      <td>89291</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>228</td>
      <td>22.16</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_merge = df_tran.merge(
    df_pf[['offer_id', 'offer_type', 'reward', 'difficulty', 'duration']],
    on='offer_id',
    how='left'
)

df_merge.sort_values(['person', 'time']).head(1)
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
      <th>time</th>
      <th>offer_id</th>
      <th>offer_type</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55972</th>
      <td>55972</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>168</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
      <td>informational</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_tran.loc[47582, 'person']
```




    '78afa995795e4d85b5d9ceeca43f5fef'




```python
person_id = '78afa995795e4d85b5d9ceeca43f5fef'

display(
    df_merge[df_merge['person'] == person_id]
    .sort_values('time')
)
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
      <th>time</th>
      <th>offer_id</th>
      <th>offer_type</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>0</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>bogo</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>15561</th>
      <td>15561</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>6</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>bogo</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>47582</th>
      <td>47582</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>132</td>
      <td>19.89</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>47583</th>
      <td>47583</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer completed</td>
      <td>132</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>bogo</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>49502</th>
      <td>49502</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>144</td>
      <td>17.78</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>53176</th>
      <td>53176</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>168</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
      <td>informational</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>85291</th>
      <td>85291</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>216</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
      <td>informational</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>87134</th>
      <td>87134</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>222</td>
      <td>19.67</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>92104</th>
      <td>92104</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>240</td>
      <td>29.72</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>141566</th>
      <td>141566</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>378</td>
      <td>23.93</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>150598</th>
      <td>150598</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>408</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
      <td>bogo</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>163375</th>
      <td>163375</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>408</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
      <td>bogo</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>201572</th>
      <td>201572</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>504</td>
      <td>f19421c1d4aa40978ebb69ca19b0e20d</td>
      <td>bogo</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>218393</th>
      <td>218393</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>510</td>
      <td>21.72</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>218394</th>
      <td>218394</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer completed</td>
      <td>510</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
      <td>bogo</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>218395</th>
      <td>218395</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer completed</td>
      <td>510</td>
      <td>f19421c1d4aa40978ebb69ca19b0e20d</td>
      <td>bogo</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>230412</th>
      <td>230412</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>transaction</td>
      <td>534</td>
      <td>26.56</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>262138</th>
      <td>262138</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer viewed</td>
      <td>582</td>
      <td>f19421c1d4aa40978ebb69ca19b0e20d</td>
      <td>bogo</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
person_id = 'f367a50b86d049799bbb0eb645ee834c'

display(
    df_merge[df_merge['person'] == person_id]
    .sort_values('time')
)
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
      <th>time</th>
      <th>offer_id</th>
      <th>offer_type</th>
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12560</th>
      <td>12560</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer received</td>
      <td>0</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>bogo</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>15540</th>
      <td>15540</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>0</td>
      <td>195.24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15541</th>
      <td>15541</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer completed</td>
      <td>0</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>bogo</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>29487</th>
      <td>29487</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer viewed</td>
      <td>42</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>bogo</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>40765</th>
      <td>40765</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>90</td>
      <td>1.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>65766</th>
      <td>65766</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer received</td>
      <td>168</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
      <td>informational</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>100241</th>
      <td>100241</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer viewed</td>
      <td>270</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
      <td>informational</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>123452</th>
      <td>123452</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer received</td>
      <td>336</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
      <td>discount</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>137409</th>
      <td>137409</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer viewed</td>
      <td>360</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
      <td>discount</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>163287</th>
      <td>163287</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer received</td>
      <td>408</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
      <td>informational</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>167593</th>
      <td>167593</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>408</td>
      <td>4.56</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>196820</th>
      <td>196820</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>480</td>
      <td>6.38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>196821</th>
      <td>196821</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer completed</td>
      <td>480</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
      <td>discount</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>214194</th>
      <td>214194</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer received</td>
      <td>504</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>bogo</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>225038</th>
      <td>225038</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>516</td>
      <td>11.77</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>225039</th>
      <td>225039</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer completed</td>
      <td>516</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>bogo</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>245105</th>
      <td>245105</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>570</td>
      <td>5.91</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>257797</th>
      <td>257797</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer received</td>
      <td>576</td>
      <td>2298d6c36e964ae4a3e7e9706d1fb8c2</td>
      <td>discount</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>268819</th>
      <td>268819</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>588</td>
      <td>197.26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>268820</th>
      <td>268820</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer completed</td>
      <td>588</td>
      <td>2298d6c36e964ae4a3e7e9706d1fb8c2</td>
      <td>discount</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>271762</th>
      <td>271762</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>offer viewed</td>
      <td>594</td>
      <td>2298d6c36e964ae4a3e7e9706d1fb8c2</td>
      <td>discount</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>274493</th>
      <td>274493</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>600</td>
      <td>21.13</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>285907</th>
      <td>285907</td>
      <td>f367a50b86d049799bbb0eb645ee834c</td>
      <td>transaction</td>
      <td>630</td>
      <td>1.56</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
first_event_time0 = (
    df_merge[df_merge['time'] == 0]
    .sort_values(['person', 'time'])
    .groupby('person')
    .first()
)

not_received_time0 = first_event_time0[first_event_time0['event'] != 'offer received']

print("time==0인데 received 아닌 사람 수:", len(not_received_time0))
display(not_received_time0[['event']].value_counts())
```

    time==0인데 received 아닌 사람 수: 111
    


    event      
    transaction    111
    Name: count, dtype: int64



```python
!jupyter nbconvert --to markdown "02_eda_v1.ipynb"
```

    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    
    

    [NbConvertApp] WARNING | pattern '01_eda_v1.ipynb' matched no files
    
