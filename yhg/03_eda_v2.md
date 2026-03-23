```python
import pandas as pd
import ast
import numpy as np


portfolio = pd.read_csv('../data/portfolio.csv')
profile = pd.read_csv('../data/profile.csv') # (17000, 6)
transcript = pd.read_csv('../data/transcript.csv') # (306534, 8)

```

# 데이터 전처리
- 결측치 처리
    - prf테이블 gender와 income에 2175개 존재
- 이상치 처리
    - age 118세 존재(2175개) / 전체적으로 높은 나이 분포
- 데이터 정제
    - became_member_on 컬럼 datetime으로 변환
    - channels 에서 각각 web, email, mobile, social로 나눌 것인가
- 데이터 파생 변수
    - value 컬럼의 딕셔너리 형태 값을 offer id, offer id + reward, amount로 나눠 파생 변수 생성



```python
# # 쓸모없는 인덱스열 삭제
# portfolio = portfolio.drop(columns=['Unnamed: 0'], errors='ignore')
# profile = profile.drop(columns=['Unnamed: 0'], errors='ignore')
# transcript = transcript.drop(columns=['Unnamed: 0'], errors='ignore')

```


```python
# 데이터 타입 date형식으로 변환
profile["became_member_on"] = pd.to_datetime(profile["became_member_on"], format="%Y%m%d")


# channels마다 파생변수 생성
portfolio['web'] = portfolio['channels'].astype(str).str.contains('web').astype(int)
portfolio['email'] = portfolio['channels'].astype(str).str.contains('email').astype(int)
portfolio['mobile'] = portfolio['channels'].astype(str).str.contains('mobile').astype(int)
portfolio['social'] = portfolio['channels'].astype(str).str.contains('social').astype(int)

# 기존 channels 컬럼 제거
portfolio = portfolio.drop('channels', axis=1)
```


```python
# 딕셔너리처럼 생긴 문자열을 진짜 딕셔너리로 변환
transcript['value'] = transcript['value'].apply(ast.literal_eval)

# 딕셔너리의 키 -> 새로운 컬럼
value_df = pd.DataFrame(transcript['value'].tolist())
transcript = pd.concat([transcript, value_df], axis=1)

# offer id를 offer_id로 컬럼명 통일
transcript['offer_id'] = transcript['offer_id'].fillna(transcript['offer id'])

# offer id 컬럼 제거
transcript = transcript.drop('offer id', axis=1)

# value 컬럼 제거
transcript = transcript.drop('value', axis=1)
```

    [NbConvertApp] Converting notebook 03_eda_v2.ipynb to markdown
    [NbConvertApp] Writing 82709 bytes to 03_eda_v2.md
    


```python
# profile의 필요없는 Unnamed:0 컬럼 제거
profile = profile.drop('Unnamed: 0', axis=1)

# transcript 기준으로 profile 데이터를 Left Join
merged_df = pd.merge(transcript, profile, left_on='person', right_on='id', how='left')

# 필요 없는 id 컬럼(person과 중복)은 버리기
merged_df = merged_df.drop(columns='id')
```


```python
display(merged_df.head())
display(merged_df.shape)
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
      <th>amount</th>
      <th>offer_id</th>
      <th>reward</th>
      <th>gender</th>
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>78afa995795e4d85b5d9ceeca43f5fef</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>NaN</td>
      <td>F</td>
      <td>75</td>
      <td>2017-05-09</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>a03223e636434f42ac4c3df47e8bac43</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>118</td>
      <td>2017-08-04</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>e2127556f4f64592b11af22de27a7932</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
      <td>NaN</td>
      <td>M</td>
      <td>68</td>
      <td>2018-04-26</td>
      <td>70000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8ec6ce2a7e7949b1bf142def7d0e0586</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>fafdcd668e3743c1bb461111dcafc2a4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>118</td>
      <td>2017-09-25</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>68617ca6246f4fbc85e91a2a49552598</td>
      <td>offer received</td>
      <td>0</td>
      <td>NaN</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>118</td>
      <td>2017-10-02</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



    (306534, 11)



```python
# # hyeong uk님의 논리를 그대로 번역한 필터링 조건
# is_118_years_old = merged_df['age'] == 118
# is_transaction = merged_df['event'] == 'transaction'

# # 두 조건을 모두 만족하는 교집합 데이터만 뽑아내기
# anomaly_transactions = merged_df[is_118_years_old & is_transaction]

# # 수상한 고객들이 긁은 총 결제 횟수와 누적 금액 확인
# total_count = anomaly_transactions['amount'].count()
# total_spent = anomaly_transactions['amount'].sum()

# print(f"118세 고객들의 총 결제 횟수: {total_count:,}건")
# print(f"118세 고객들의 총 결제 금액: ${total_spent:,.2f}")
```


```python
# 결측치 처리

# gender의 결측치 'Unknown'으로 채우기 
merged_df['gender'] = merged_df['gender'].fillna('Unknown')


# age의 118을 결측치(NaN)로 바꿔주기 
merged_df['age'] = merged_df['age'].replace(118, np.nan)
# income은 이미 결측치(NaN) 상태
```


```python
# # gender 필터링
# is_gender_others = merged_df['gender'] == 'O'

# # 두 조건을 모두 만족하는 교집합 데이터만 뽑아내기
# anomaly_transactions2 = merged_df[is_gender_others & is_transaction]

# # 수상한 고객들이 긁은 총 결제 횟수와 누적 금액 확인
# total_count2 = anomaly_transactions2['amount'].count()
# total_spent2 = anomaly_transactions2['amount'].sum()

# print(f"others 고객들의 총 결제 횟수: {total_count2:,}건")
# print(f"others 고객들의 총 결제 금액: ${total_spent2:,.2f}")
```


```python
# portfolio 테이블도 병합

# portfolio 테이블의 필요없는 인덱스 컬럼 제거
portfolio = portfolio.drop('Unnamed: 0', axis=1)

all_merge_df = pd.merge(
    merged_df,
    portfolio,
    left_on='offer_id',
    right_on='id',
    how='left'
)

all_merge_df = all_merge_df.drop(columns='id')

# reward 컬럼명 변경(명확하게)
all_merge_df = all_merge_df.rename(columns={
    "reward_x": "transcript_reward",
    "reward_y": "portfolio_reward"
})
```


```python
# offer_id 이름 변경 (쿠폰명_difficulty_reward_duration)
portfolio['offer_name'] = (
    portfolio['offer_type'] + '_' + 
    portfolio['difficulty'].astype(str) + '_' + 
    portfolio['reward'].astype(str) + '_' + 
    portfolio['duration'].astype(str)
)
# id : key, offer_name : value
offer_name_dict = portfolio.set_index('id')['offer_name'].to_dict()
all_merge_df['offer_id'] = all_merge_df['offer_id'].map(offer_name_dict)


# 사람(person)별로 먼저 묶고, 그 안에서 시간(time) 순서대로 오름차순 정렬
all_merge_df = all_merge_df.sort_values(by=['person', 'time', 'Unnamed: 0']) # - Unnamed: 0 순서 추가
```


```python
all_merge_df
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
      <th>amount</th>
      <th>offer_id</th>
      <th>transcript_reward</th>
      <th>gender</th>
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
      <th>portfolio_reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>web</th>
      <th>email</th>
      <th>mobile</th>
      <th>social</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55972</th>
      <td>55972</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>168</td>
      <td>NaN</td>
      <td>informational_0_0_3</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>informational</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>77705</th>
      <td>77705</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>192</td>
      <td>NaN</td>
      <td>informational_0_0_3</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>informational</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>89291</th>
      <td>89291</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>228</td>
      <td>22.16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>113605</th>
      <td>113605</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>336</td>
      <td>NaN</td>
      <td>informational_0_0_4</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>139992</th>
      <td>139992</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>372</td>
      <td>NaN</td>
      <td>informational_0_0_4</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>258361</th>
      <td>258361</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>transaction</td>
      <td>576</td>
      <td>14.23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>258362</th>
      <td>258362</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>offer completed</td>
      <td>576</td>
      <td>NaN</td>
      <td>discount_10_2_7</td>
      <td>2.0</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>discount</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>262475</th>
      <td>262475</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>offer viewed</td>
      <td>582</td>
      <td>NaN</td>
      <td>discount_10_2_7</td>
      <td>NaN</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>discount</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>274809</th>
      <td>274809</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>transaction</td>
      <td>606</td>
      <td>10.12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>289924</th>
      <td>289924</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>transaction</td>
      <td>648</td>
      <td>18.91</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>306534 rows × 19 columns</p>
</div>




```python
# 무엇을 위해 하는 코드인가? -> informational이 아닌 completed와 amount 경우만 선택하는 과정

# 조건 1: 쿠폰 타입이 bogo 이거나(in) discount 인 것
cond_offers = all_merge_df['offer_type'].isin(['bogo', 'discount'])

# 조건 2: 이벤트 종류가 transaction(결제) 인 것
cond_transactions = all_merge_df['event'] == 'transaction'

# 위 두 조건 중 하나라도 만족하는(|) 데이터만 쏙 뽑아서 덮어씌우기
target_df = all_merge_df[cond_offers | cond_transactions].copy()

# 잘 걸러졌는지 눈으로 확인해보기
print(target_df['offer_type'].value_counts(dropna=False))
print(target_df['event'].value_counts(dropna=False))
display(target_df.head())
display(target_df.shape)
```

    offer_type
    NaN         138953
    bogo         71617
    discount     69898
    Name: count, dtype: int64
    event
    transaction        138953
    offer received      61042
    offer viewed        46894
    offer completed     33579
    Name: count, dtype: int64
    


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
      <th>amount</th>
      <th>offer_id</th>
      <th>transcript_reward</th>
      <th>gender</th>
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
      <th>portfolio_reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>web</th>
      <th>email</th>
      <th>mobile</th>
      <th>social</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89291</th>
      <td>89291</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>228</td>
      <td>22.16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>153401</th>
      <td>153401</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>408</td>
      <td>NaN</td>
      <td>bogo_5_5_5</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>168412</th>
      <td>168412</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>414</td>
      <td>8.57</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>168413</th>
      <td>168413</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer completed</td>
      <td>414</td>
      <td>NaN</td>
      <td>bogo_5_5_5</td>
      <td>5.0</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>187554</th>
      <td>187554</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>456</td>
      <td>NaN</td>
      <td>bogo_5_5_5</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>bogo</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



    (280468, 19)



```python
# person당 offer_id를 하나의 행으로 설정하여, 흩어진 고객 행동의 순서를 보기 편하게 해주는 "피벗테이블 생성 코드"

# 1. 피벗을 돌릴 '쿠폰 이력서' 데이터만 빼내기
offers_df = target_df[target_df['event'] != 'transaction'].copy()

# 2. 안전한 금고에 보관할 '순수 영수증' 데이터만 빼내기
transactions_df = target_df[target_df['event'] == 'transaction'].copy()

print(f"피벗할 쿠폰 데이터: {len(offers_df)} 줄")
print(f"금고에 보관한 영수증: {len(transactions_df)} 줄")
```

    피벗할 쿠폰 데이터: 141515 줄
    금고에 보관한 영수증: 138953 줄
    


```python
# 시간 순으로 정렬 후, 사람&쿠폰별 첫 번째 이벤트만 쏙 뽑아오기
first_events = offers_df.sort_values(['person', 'offer_id', 'time']).groupby(['person', 'offer_id'])['event'].first()

# 그 첫 번째 이벤트가 'received'가 아닌 놈들(유령 데이터)만 필터링
ghost_data = first_events[first_events != 'offer received']

print(f"🚨 received 없이 시작하는 유령 세션 수: {len(ghost_data)}개")
if len(ghost_data) > 0:
    display(ghost_data.head())
```

    🚨 received 없이 시작하는 유령 세션 수: 0개
    


```python
# 1. 시간 순서대로 예쁘게 줄 세우기 (시간이 꼬이면 안 되니까 필수)
offers_df = offers_df.sort_values(['person', 'offer_id', 'time'])

# 2. 'received' 이벤트가 등장할 때마다 1, 아니면 0인 깃발(Flag) 만들기
offers_df['is_received'] = (offers_df['event'] == 'offer received').astype(int)

# 3. [마법의 함수] 사람과 쿠폰 단위로 묶어서, 깃발을 누적해서 더하기 (Cumsum)
offers_df['offer_cycle'] = offers_df.groupby(['person', 'offer_id'])['is_received'].cumsum()

# 4. 이제 찝찝함 없이 피벗 돌리기 (기준점에 offer_cycle 추가!)
pivot_df = offers_df.pivot_table(
    index=['person', 'offer_id', 'offer_cycle'],  # "누구의 / 어떤 쿠폰의 / 몇 회차인가?"
    columns='event',
    values='time',
    aggfunc='min'  # 이제 한 회차 안에는 중복이 없으니 min을 써도 아무 왜곡이 안 일어납니다
).reset_index()

# 깔끔하게 정리
pivot_df.columns.name = None
pivot_df = pivot_df[['person', 'offer_id', 'offer_cycle', 'offer received', 'offer viewed', 'offer completed']]

display(pivot_df.head())
display(pivot_df.shape)
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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>bogo_5_5_5</td>
      <td>1</td>
      <td>408.0</td>
      <td>456.0</td>
      <td>414.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_10</td>
      <td>1</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>528.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_7</td>
      <td>1</td>
      <td>576.0</td>
      <td>NaN</td>
      <td>576.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>1</td>
      <td>168.0</td>
      <td>216.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>2</td>
      <td>576.0</td>
      <td>630.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



    (61042, 6)



```python
# 1. 원본에서 offer_id와 offer_type 짝꿍 사전 만들기
offer_dict = offers_df[['offer_id', 'offer_type']].drop_duplicates().set_index('offer_id')['offer_type'].to_dict()

# 2. 피벗 테이블의 offer_id를 보고, 임시로 쿠폰 타입(bogo, discount)을 가져오기
temp_offer_type = pivot_df['offer_id'].map(offer_dict)

# 3. [핵심] 기존 숫자였던 'offer_cycle' 컬럼 위에 곧바로 덮어쓰기! 
pivot_df['offer_cycle'] = temp_offer_type.str.capitalize() + '_' + pivot_df['offer_cycle'].astype(str)

```


```python
# 결과 확인
display(pivot_df.head())
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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>456.0</td>
      <td>414.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>528.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>NaN</td>
      <td>576.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>168.0</td>
      <td>216.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_2</td>
      <td>576.0</td>
      <td>630.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 피벗테이블에 amount 붙이기

# 1. 금고(transactions_df)에서 영수증 알맹이만 꺼내기
transactions_df = transactions_df[['person', 'time', 'amount']]

# 2. 피벗 테이블(pivot_df)에 영수증(receipts) 1:1 도킹하기!
final_df = pivot_df.merge(
    transactions_df,
    left_on=['person', 'offer completed'],  # 왼쪽 표(피벗)의 도킹 기준: "누구의 / 언제 달성(completed)한 쿠폰인가?"
    right_on=['person', 'time'],      # 오른쪽 표(영수증)의 도킹 기준: "누가 / 언제(time) 결제했는가?"
    how='left'                        # 조인 방식: "피벗 표를 기준으로 하고, 영수증이 없으면 빈칸(NaN)으로 둬라!"
)

# 3. 도킹 끝나고 쓸모없어진 'time' 기둥 버리기
final_df = final_df.drop(columns=['time'])

# 4. 가슴이 웅장해지는 최종 결과물 확인!
display(final_df.head())
display(final_df.shape)
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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>456.0</td>
      <td>414.0</td>
      <td>8.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>528.0</td>
      <td>14.11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>NaN</td>
      <td>576.0</td>
      <td>10.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>168.0</td>
      <td>216.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_2</td>
      <td>576.0</td>
      <td>630.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



    (61042, 7)


---

## 여기까지 괜찮지만, 총 매출을 구할때 더블카운팅 문제가 발생함. 어떻게 해결할래?
### 일단 더블카운팅 찾기


```python
# 한번의 결제에 두개 이상의 completed가 발생한 경우를 찾기

# person과 offer completed가 똑같은 모든 중복 행(NaN 제외)
dup_mask = final_df.duplicated(subset=['person', 'offer completed'], keep=False) & final_df['offer completed'].notna()
problem_df = final_df[dup_mask]

# 정렬
problem_df = problem_df.sort_values(by=['person', 'offer completed', 'offer_id', 'offer_cycle'])

display(problem_df.head(20))
print(len(problem_df))
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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>22.05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>432.0</td>
      <td>576.0</td>
      <td>22.05</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0020c2b971eb4e9188eac86d93036a77</td>
      <td>bogo_10_10_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>426.0</td>
      <td>510.0</td>
      <td>17.24</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0020c2b971eb4e9188eac86d93036a77</td>
      <td>discount_10_2_10</td>
      <td>Discount_2</td>
      <td>336.0</td>
      <td>NaN</td>
      <td>510.0</td>
      <td>17.24</td>
    </tr>
    <tr>
      <th>91</th>
      <td>00ae03011f9f49b8a4b3e6d416678b0b</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_2</td>
      <td>504.0</td>
      <td>534.0</td>
      <td>618.0</td>
      <td>30.83</td>
    </tr>
    <tr>
      <th>94</th>
      <td>00ae03011f9f49b8a4b3e6d416678b0b</td>
      <td>discount_7_3_7</td>
      <td>Discount_2</td>
      <td>576.0</td>
      <td>606.0</td>
      <td>618.0</td>
      <td>30.83</td>
    </tr>
    <tr>
      <th>133</th>
      <td>00c2f812f4604c8893152a5c6572030e</td>
      <td>bogo_10_10_5</td>
      <td>Bogo_3</td>
      <td>576.0</td>
      <td>600.0</td>
      <td>582.0</td>
      <td>24.21</td>
    </tr>
    <tr>
      <th>135</th>
      <td>00c2f812f4604c8893152a5c6572030e</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>NaN</td>
      <td>582.0</td>
      <td>24.21</td>
    </tr>
    <tr>
      <th>157</th>
      <td>00cf1bbec83f4a658f8994e556db4633</td>
      <td>discount_10_2_10</td>
      <td>Discount_2</td>
      <td>336.0</td>
      <td>438.0</td>
      <td>564.0</td>
      <td>33.28</td>
    </tr>
    <tr>
      <th>158</th>
      <td>00cf1bbec83f4a658f8994e556db4633</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>564.0</td>
      <td>33.28</td>
    </tr>
    <tr>
      <th>169</th>
      <td>00d791e20c564add8056498e40eb56cc</td>
      <td>bogo_10_10_5</td>
      <td>Bogo_1</td>
      <td>576.0</td>
      <td>606.0</td>
      <td>582.0</td>
      <td>10.38</td>
    </tr>
    <tr>
      <th>171</th>
      <td>00d791e20c564add8056498e40eb56cc</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>510.0</td>
      <td>582.0</td>
      <td>10.38</td>
    </tr>
    <tr>
      <th>174</th>
      <td>00d7c95f793a4212af44e632fdc1e431</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>498.0</td>
      <td>504.0</td>
      <td>18.58</td>
    </tr>
    <tr>
      <th>177</th>
      <td>00d7c95f793a4212af44e632fdc1e431</td>
      <td>discount_10_2_7</td>
      <td>Discount_2</td>
      <td>504.0</td>
      <td>NaN</td>
      <td>504.0</td>
      <td>18.58</td>
    </tr>
    <tr>
      <th>197</th>
      <td>00ed7e22b32749cfafbfd88592d401d4</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>522.0</td>
      <td>14.64</td>
    </tr>
    <tr>
      <th>198</th>
      <td>00ed7e22b32749cfafbfd88592d401d4</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>NaN</td>
      <td>522.0</td>
      <td>14.64</td>
    </tr>
    <tr>
      <th>217</th>
      <td>0103de989e084e0fab400e80678d7591</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>336.0</td>
      <td>384.0</td>
      <td>384.0</td>
      <td>20.20</td>
    </tr>
    <tr>
      <th>218</th>
      <td>0103de989e084e0fab400e80678d7591</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>384.0</td>
      <td>20.20</td>
    </tr>
    <tr>
      <th>216</th>
      <td>0103de989e084e0fab400e80678d7591</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_2</td>
      <td>576.0</td>
      <td>588.0</td>
      <td>588.0</td>
      <td>9.00</td>
    </tr>
    <tr>
      <th>219</th>
      <td>0103de989e084e0fab400e80678d7591</td>
      <td>discount_20_5_10</td>
      <td>Discount_2</td>
      <td>504.0</td>
      <td>NaN</td>
      <td>588.0</td>
      <td>9.00</td>
    </tr>
  </tbody>
</table>
</div>


    5056
    


```python
# 만약, 동일한 completed에 다른 amount를 가진다면 그것은 다른 transaction임. => 더블카운팅이 아님

# person과 offer completed로 묶어서, amount가 다른 행이 몇개인지 확인해보는 코드
amount_types = problem_df.groupby(['person', 'offer completed'])['amount'].nunique()
suspects = amount_types[amount_types > 1]

print(f"금액이 다르게 찍힌 중복 케이스: {len(suspects)}")
```

    금액이 다르게 찍힌 중복 케이스: 0
    

### 그럼 최종적으로 더블카운팅 찾는 문제에서 놓친 경우는 completed 이후에 같은 시간내에 또 received->completed까지 간 경우
- ex) person A가 bogo_5_5_7을 amount 20으로, 500에 offer completed했을 때 -> 동일한 사람이 동일 amount로 500 안에서 received하고 completed까지 이어진 경우


```python
# '즉시 달성'/'정상 달성' 구분
problem_df['is_instant'] = problem_df['offer received'] == problem_df['offer completed']
problem_df['is_normal'] = problem_df['offer received'] < problem_df['offer completed']

# person과 offer completed 기준으로 묶어서'즉시 달성'과 '정상 달성'이 모두 섞여 있는지 확인
group_flags = problem_df.groupby(['person', 'offer completed']).agg(
    has_instant=('is_instant', 'any'),
    has_normal=('is_normal', 'any')
)

# 정상 달성과 즉시 달성이 한 시간에 같이 일어난 타겟 그룹의 인덱스 뽑기
target_groups = group_flags[group_flags['has_instant'] & group_flags['has_normal']].index

# problem_df에서 이 타겟 그룹에 해당하는 애들만 걸러내기
# (set_index를 써서 튜플 형태의 인덱스와 빠르게 매칭)
result_df = problem_df[problem_df.set_index(['person', 'offer completed']).index.isin(target_groups)]
result_df = result_df.sort_values(by=['person', 'offer completed', 'offer received']).drop(columns=['is_instant', 'is_normal'])


print(f"[정상 달성 + 즉시 달성]이 동시에 일어난 타겟 건수: {len(result_df)}건")
display(result_df[['person', 'offer_id', 'offer_cycle', 'offer received', 'offer viewed', 'offer completed', 'amount']].head(20))

print(len(result_df[result_df['offer received'] == result_df['offer completed']]))
```

    [정상 달성 + 즉시 달성]이 동시에 일어난 타겟 건수: 633건
    


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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>174</th>
      <td>00d7c95f793a4212af44e632fdc1e431</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>498.0</td>
      <td>504.0</td>
      <td>18.58</td>
    </tr>
    <tr>
      <th>177</th>
      <td>00d7c95f793a4212af44e632fdc1e431</td>
      <td>discount_10_2_7</td>
      <td>Discount_2</td>
      <td>504.0</td>
      <td>NaN</td>
      <td>504.0</td>
      <td>18.58</td>
    </tr>
    <tr>
      <th>474</th>
      <td>021adce38ab34ede96422ae107643fd5</td>
      <td>discount_10_2_7</td>
      <td>Discount_3</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>15.64</td>
    </tr>
    <tr>
      <th>475</th>
      <td>021adce38ab34ede96422ae107643fd5</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>576.0</td>
      <td>576.0</td>
      <td>15.64</td>
    </tr>
    <tr>
      <th>595</th>
      <td>02abd909ebc94aca8766f3f0ee39db80</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>336.0</td>
      <td>396.0</td>
      <td>408.0</td>
      <td>14.80</td>
    </tr>
    <tr>
      <th>596</th>
      <td>02abd909ebc94aca8766f3f0ee39db80</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>414.0</td>
      <td>408.0</td>
      <td>14.80</td>
    </tr>
    <tr>
      <th>618</th>
      <td>02c89861ce2c4010bf4ed63f6f6d5df3</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>576.0</td>
      <td>19.54</td>
    </tr>
    <tr>
      <th>617</th>
      <td>02c89861ce2c4010bf4ed63f6f6d5df3</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>582.0</td>
      <td>576.0</td>
      <td>19.54</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>04e39b3e8fc449cfbcafd7dba2429f1c</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>0.0</td>
      <td>48.0</td>
      <td>168.0</td>
      <td>20.71</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>04e39b3e8fc449cfbcafd7dba2429f1c</td>
      <td>bogo_10_10_5</td>
      <td>Bogo_1</td>
      <td>168.0</td>
      <td>168.0</td>
      <td>168.0</td>
      <td>20.71</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>051e221bbd75481c8d935f6adcee2edc</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>NaN</td>
      <td>576.0</td>
      <td>16.93</td>
    </tr>
    <tr>
      <th>1175</th>
      <td>051e221bbd75481c8d935f6adcee2edc</td>
      <td>discount_10_2_10</td>
      <td>Discount_3</td>
      <td>576.0</td>
      <td>612.0</td>
      <td>576.0</td>
      <td>16.93</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>07805efd53494b8fa08f661e311dad9e</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>486.0</td>
      <td>504.0</td>
      <td>31.50</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>07805efd53494b8fa08f661e311dad9e</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>NaN</td>
      <td>504.0</td>
      <td>31.50</td>
    </tr>
    <tr>
      <th>1782</th>
      <td>07d5af069ead41e9994707b6caf0312a</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>168.0</td>
      <td>168.0</td>
      <td>336.0</td>
      <td>56.77</td>
    </tr>
    <tr>
      <th>1785</th>
      <td>07d5af069ead41e9994707b6caf0312a</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>336.0</td>
      <td>336.0</td>
      <td>336.0</td>
      <td>56.77</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>08926f1f08c34a4a9681ab3d6a5ef17e</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>414.0</td>
      <td>576.0</td>
      <td>37.01</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>08926f1f08c34a4a9681ab3d6a5ef17e</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>37.01</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>08926f1f08c34a4a9681ab3d6a5ef17e</td>
      <td>discount_10_2_7</td>
      <td>Discount_2</td>
      <td>576.0</td>
      <td>612.0</td>
      <td>576.0</td>
      <td>37.01</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>089884c069654beba55de2442e32ea82</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>168.0</td>
      <td>186.0</td>
      <td>336.0</td>
      <td>19.23</td>
    </tr>
  </tbody>
</table>
</div>


    308
    


```python
target_person = "0011e0d4e6b944f998e987f904e8c1e5"

person_df = result_df[result_df['person'] == target_person]

display(person_df.sort_values(by=['offer completed', 'offer received']))
print(f"건수: {len(person_df)}")
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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


    건수: 0
    


```python
# 1. 즉시 달성 / 정상 달성 구분
problem_df['is_instant'] = problem_df['offer received'] == problem_df['offer completed']

problem_df['is_normal'] = (
    (problem_df['offer received'] <= problem_df['offer viewed']) &
    (problem_df['offer viewed'] <= problem_df['offer completed']) &
    (problem_df['offer received'] < problem_df['offer completed'])
)

# 2. person + offer completed 기준으로 그룹 집계
group_flags = problem_df.groupby(['person', 'offer completed']).agg(
    instant_cnt=('is_instant', 'sum'),
    normal_cnt=('is_normal', 'sum')
).reset_index()

# 3. 케이스 유형 분류
group_flags['case_type'] = None

# 즉시 + 정상 혼합
group_flags.loc[
    (group_flags['instant_cnt'] >= 1) & (group_flags['normal_cnt'] >= 1),
    'case_type'
] = 'mixed_instant_normal'

# 정상 + 정상 중복
group_flags.loc[
    (group_flags['normal_cnt'] >= 2) & (group_flags['case_type'].isna()),
    'case_type'
] = 'normal_double'

# 둘 다 해당하는 경우
group_flags.loc[
    (group_flags['instant_cnt'] >= 1) & (group_flags['normal_cnt'] >= 2),
    'case_type'
] = 'mixed_and_normal_double'

# 4. 이상 케이스 그룹만 추출
target_groups = group_flags[group_flags['case_type'].notna()][
    ['person', 'offer completed', 'case_type', 'instant_cnt', 'normal_cnt']
]

# 5. 원본 데이터와 merge해서 해당 그룹의 상세 행만 보기
result_df = problem_df.merge(
    target_groups,
    on=['person', 'offer completed'],
    how='inner'
).sort_values(
    by=['person', 'offer completed', 'offer received']
).drop(columns=['is_instant', 'is_normal'])

# 6. 전체 결과 확인
print(f"이상 케이스 전체 건수: {len(result_df)}건")
display(
    result_df[
        ['person', 'offer_id', 'offer_cycle', 'offer received', 'offer viewed',
         'offer completed', 'amount', 'case_type', 'instant_cnt', 'normal_cnt']
    ].head(30)
)

# 7. 케이스 유형별 그룹 수 확인
print("\n케이스 유형별 그룹 수")
display(target_groups['case_type'].value_counts())
```

    이상 케이스 전체 건수: 2209건
    


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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>amount</th>
      <th>case_type</th>
      <th>instant_cnt</th>
      <th>normal_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>432.0</td>
      <td>576.0</td>
      <td>22.05</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>22.05</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00ae03011f9f49b8a4b3e6d416678b0b</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_2</td>
      <td>504.0</td>
      <td>534.0</td>
      <td>618.0</td>
      <td>30.83</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00ae03011f9f49b8a4b3e6d416678b0b</td>
      <td>discount_7_3_7</td>
      <td>Discount_2</td>
      <td>576.0</td>
      <td>606.0</td>
      <td>618.0</td>
      <td>30.83</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00cf1bbec83f4a658f8994e556db4633</td>
      <td>discount_10_2_10</td>
      <td>Discount_2</td>
      <td>336.0</td>
      <td>438.0</td>
      <td>564.0</td>
      <td>33.28</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>00cf1bbec83f4a658f8994e556db4633</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>564.0</td>
      <td>33.28</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>00d7c95f793a4212af44e632fdc1e431</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>498.0</td>
      <td>504.0</td>
      <td>18.58</td>
      <td>mixed_instant_normal</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>00d7c95f793a4212af44e632fdc1e431</td>
      <td>discount_10_2_7</td>
      <td>Discount_2</td>
      <td>504.0</td>
      <td>NaN</td>
      <td>504.0</td>
      <td>18.58</td>
      <td>mixed_instant_normal</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>015c3d28c67e46aa95e9ec97c27220e8</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>510.0</td>
      <td>618.0</td>
      <td>22.12</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>015c3d28c67e46aa95e9ec97c27220e8</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>576.0</td>
      <td>582.0</td>
      <td>618.0</td>
      <td>22.12</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>01633b71b3a2457aa7d35d8bcc3afb5a</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>552.0</td>
      <td>600.0</td>
      <td>24.76</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>01633b71b3a2457aa7d35d8bcc3afb5a</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>594.0</td>
      <td>600.0</td>
      <td>24.76</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>01784d3e205548a594ba3fcdbdaaf17d</td>
      <td>bogo_10_10_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>444.0</td>
      <td>516.0</td>
      <td>24.67</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>01784d3e205548a594ba3fcdbdaaf17d</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>516.0</td>
      <td>24.67</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>01887dcd32b64feb807e2436548b6c87</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>408.0</td>
      <td>570.0</td>
      <td>25.04</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>01887dcd32b64feb807e2436548b6c87</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>522.0</td>
      <td>570.0</td>
      <td>25.04</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>01b6d7e8f0884deb936a8a7f15dba895</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>420.0</td>
      <td>558.0</td>
      <td>12.78</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>01b6d7e8f0884deb936a8a7f15dba895</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>510.0</td>
      <td>558.0</td>
      <td>12.78</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0215efe5136d4a038cb81eae92d59368</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>336.0</td>
      <td>342.0</td>
      <td>534.0</td>
      <td>11.96</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0215efe5136d4a038cb81eae92d59368</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>528.0</td>
      <td>534.0</td>
      <td>11.96</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>021adce38ab34ede96422ae107643fd5</td>
      <td>discount_10_2_7</td>
      <td>Discount_3</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>15.64</td>
      <td>mixed_instant_normal</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>021adce38ab34ede96422ae107643fd5</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>576.0</td>
      <td>576.0</td>
      <td>15.64</td>
      <td>mixed_instant_normal</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>021c1940868647efbcb40ccdb942813b</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>168.0</td>
      <td>192.0</td>
      <td>366.0</td>
      <td>15.49</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>021c1940868647efbcb40ccdb942813b</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>336.0</td>
      <td>348.0</td>
      <td>366.0</td>
      <td>15.49</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>021c1940868647efbcb40ccdb942813b</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_2</td>
      <td>504.0</td>
      <td>528.0</td>
      <td>624.0</td>
      <td>19.03</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>021c1940868647efbcb40ccdb942813b</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>576.0</td>
      <td>594.0</td>
      <td>624.0</td>
      <td>19.03</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>02a458e1233342b79caff81edbcc30a9</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>522.0</td>
      <td>618.0</td>
      <td>21.51</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>02a458e1233342b79caff81edbcc30a9</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>576.0</td>
      <td>588.0</td>
      <td>618.0</td>
      <td>21.51</td>
      <td>normal_double</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>02abd909ebc94aca8766f3f0ee39db80</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>336.0</td>
      <td>396.0</td>
      <td>408.0</td>
      <td>14.80</td>
      <td>mixed_instant_normal</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>02abd909ebc94aca8766f3f0ee39db80</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>414.0</td>
      <td>408.0</td>
      <td>14.80</td>
      <td>mixed_instant_normal</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    
    케이스 유형별 그룹 수
    


    case_type
    normal_double              861
    mixed_instant_normal       216
    mixed_and_normal_double     10
    Name: count, dtype: int64


- 더블 카운팅이 아닐 수도 있다 -> 만약 결제를 2번 했는데 우연히 가격이 같은 상품을 구매해서 포함된거라면?


```python
import pandas as pd
import numpy as np

# =========================================================
# 0. 데이터 복사
# =========================================================
df = problem_df.copy()
ts = transcript.copy()

# =========================================================
# 1. 즉시 달성 / 정상 달성 정의
# =========================================================
df['is_instant'] = df['offer received'] == df['offer completed']

df['is_normal'] = (
    (df['offer received'] <= df['offer viewed']) &
    (df['offer viewed'] <= df['offer completed']) &
    (df['offer received'] < df['offer completed'])
)

# =========================================================
# 2. 더블 카운팅 "후보군" (이상 케이스) 추출
# =========================================================
group_flags = df.groupby(['person', 'offer completed']).agg(
    instant_cnt=('is_instant', 'sum'),
    normal_cnt=('is_normal', 'sum'),
    completed_row_cnt=('offer_id', 'size'),
    unique_offer_cnt=('offer_id', 'nunique')
).reset_index()

group_flags['has_mixed'] = (group_flags['instant_cnt'] >= 1) & (group_flags['normal_cnt'] >= 1)
group_flags['has_normal_double'] = group_flags['normal_cnt'] >= 2

candidate_groups = group_flags[
    group_flags['has_mixed'] | group_flags['has_normal_double']
].copy()

# =========================================================
# 3. transaction 정보 집계
# =========================================================
txn_df = ts[ts['event'] == 'transaction'].copy()

txn_counts = txn_df.groupby(['person', 'time']).agg(
    txn_cnt=('event', 'size'),
    txn_amount_sum=('amount', 'sum')
).reset_index().rename(columns={'time': 'offer completed'})

# =========================================================
# 4. 후보군 상세 데이터 추출
# =========================================================
candidate_detail = df.merge(
    candidate_groups[['person', 'offer completed']],
    on=['person', 'offer completed'],
    how='inner'
)

# =========================================================
# 5. 같은 offer_id 중복 여부 확인
# =========================================================
offer_dup = candidate_detail.groupby(
    ['person', 'offer completed', 'offer_id']
).size().reset_index(name='same_offer_cnt')

same_offer_dup = offer_dup.groupby(['person', 'offer completed']).agg(
    max_same_offer_cnt=('same_offer_cnt', 'max'),
    duplicated_offer_id_cnt=('same_offer_cnt', lambda x: (x >= 2).sum())
).reset_index()

# =========================================================
# 6. 그룹 기준 통합
# =========================================================
review_df = candidate_groups.merge(
    txn_counts,
    on=['person', 'offer completed'],
    how='left'
).merge(
    same_offer_dup,
    on=['person', 'offer completed'],
    how='left'
)

review_df['txn_cnt'] = review_df['txn_cnt'].fillna(0).astype(int)
review_df['txn_amount_sum'] = review_df['txn_amount_sum'].fillna(0)
review_df['max_same_offer_cnt'] = review_df['max_same_offer_cnt'].fillna(1).astype(int)
review_df['duplicated_offer_id_cnt'] = review_df['duplicated_offer_id_cnt'].fillna(0).astype(int)

# =========================================================
# 7. "더블 카운팅이 아닐 가능성 높은 데이터" 제거
# =========================================================
filtered_groups = review_df[
    ~(
        # transaction 충분 + offer 중복 없음 → 정상 가능성 높음
        (review_df['txn_cnt'] >= review_df['completed_row_cnt']) &
        (review_df['unique_offer_cnt'] == review_df['completed_row_cnt'])
    )
].copy()

# =========================================================
# 8. case_type 부여 (설명용)
# =========================================================
filtered_groups['case_type'] = 'other'

filtered_groups.loc[
    filtered_groups['has_mixed'] & ~filtered_groups['has_normal_double'],
    'case_type'
] = 'mixed_instant_normal'

filtered_groups.loc[
    ~filtered_groups['has_mixed'] & filtered_groups['has_normal_double'],
    'case_type'
] = 'normal_double'

filtered_groups.loc[
    filtered_groups['has_mixed'] & filtered_groups['has_normal_double'],
    'case_type'
] = 'mixed_and_normal_double'

# =========================================================
# 9. 상세 데이터 최종 생성
# =========================================================
final_df = candidate_detail.merge(
    filtered_groups[
        [
            'person', 'offer completed',
            'instant_cnt', 'normal_cnt',
            'completed_row_cnt', 'unique_offer_cnt',
            'txn_cnt', 'txn_amount_sum',
            'max_same_offer_cnt', 'duplicated_offer_id_cnt',
            'case_type'
        ]
    ],
    on=['person', 'offer completed'],
    how='inner'
).sort_values(['person', 'offer completed', 'offer received'])

# 불필요 컬럼 제거
final_df = final_df.drop(columns=['is_instant', 'is_normal'])

# =========================================================
# 10. 결과 출력
# =========================================================
print("=== 전체 후보 그룹 수 (필터링 전) ===")
print(len(review_df))

print("\n=== 필터링 후 남은 그룹 수 ===")
print(len(filtered_groups))

print("\n=== case_type 분포 ===")
display(filtered_groups['case_type'].value_counts())

print("\n=== 상위 더블 카운팅 후보 그룹 ===")
display(
    filtered_groups.sort_values(
        ['completed_row_cnt', 'txn_cnt', 'duplicated_offer_id_cnt'],
        ascending=[False, True, False]
    ).head(20)
)

print("\n=== 최종 상세 데이터 ===")
display(
    final_df[
        [
            'person', 'offer_id', 'offer_cycle',
            'offer received', 'offer viewed', 'offer completed', 'amount',
            'case_type',
            'completed_row_cnt', 'txn_cnt', 'txn_amount_sum',
            'unique_offer_cnt', 'duplicated_offer_id_cnt'
        ]
    ].head(30)
)
```

    === 전체 후보 그룹 수 (필터링 전) ===
    1087
    
    === 필터링 후 남은 그룹 수 ===
    1087
    
    === case_type 분포 ===
    


    case_type
    normal_double              861
    mixed_instant_normal       216
    mixed_and_normal_double     10
    Name: count, dtype: int64


    
    === 상위 더블 카운팅 후보 그룹 ===
    


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
      <th>person</th>
      <th>offer completed</th>
      <th>instant_cnt</th>
      <th>normal_cnt</th>
      <th>completed_row_cnt</th>
      <th>unique_offer_cnt</th>
      <th>has_mixed</th>
      <th>has_normal_double</th>
      <th>txn_cnt</th>
      <th>txn_amount_sum</th>
      <th>max_same_offer_cnt</th>
      <th>duplicated_offer_id_cnt</th>
      <th>case_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>504</th>
      <td>75bb371cf36d4a9186397a9866ed2fbe</td>
      <td>576.0</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>31.44</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_and_normal_double</td>
    </tr>
    <tr>
      <th>37</th>
      <td>088debab4050432abc4817f192abc702</td>
      <td>516.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>25.59</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>38</th>
      <td>08926f1f08c34a4a9681ab3d6a5ef17e</td>
      <td>576.0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>37.01</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_and_normal_double</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0999506e6713408b96827816f547f0ff</td>
      <td>606.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>12.93</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>118</th>
      <td>1fc1b2cb8a7d4fc8966c12a56aea8585</td>
      <td>504.0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>23.47</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_and_normal_double</td>
    </tr>
    <tr>
      <th>133</th>
      <td>24b7186006df42968b948016dda0030a</td>
      <td>576.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>False</td>
      <td>1</td>
      <td>24.25</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_instant_normal</td>
    </tr>
    <tr>
      <th>145</th>
      <td>2698c067b1d74815bc3a899e301f93e8</td>
      <td>558.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>16.24</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>175</th>
      <td>2d40e25b82b94b37bbad2332e90a1034</td>
      <td>510.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>23.59</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>183</th>
      <td>2e60dab0d4ce46faa0cd690ac315c440</td>
      <td>648.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>24.21</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>194</th>
      <td>2fc5fa0b50f944e398b903b0be851678</td>
      <td>504.0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>510.52</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_and_normal_double</td>
    </tr>
    <tr>
      <th>240</th>
      <td>393e6938e7eb46cb86bcb04f231efc0f</td>
      <td>576.0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>24.44</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_and_normal_double</td>
    </tr>
    <tr>
      <th>324</th>
      <td>4cbe33c601a5407f8202086565c55111</td>
      <td>558.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>31.72</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>345</th>
      <td>528339af535e43aaaa128fc5044fb46d</td>
      <td>600.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>21.04</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>374</th>
      <td>59d01620b8c04fdbb630040c6deeb0a1</td>
      <td>510.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>11.30</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>406</th>
      <td>609dffa9eb714e3281a9aca1e5ce48f0</td>
      <td>504.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>False</td>
      <td>1</td>
      <td>21.46</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_instant_normal</td>
    </tr>
    <tr>
      <th>413</th>
      <td>631b3ebd660243df8069c49d4c37f836</td>
      <td>504.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>False</td>
      <td>1</td>
      <td>84.76</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_instant_normal</td>
    </tr>
    <tr>
      <th>415</th>
      <td>63ee1bb377864f1481f45819e3c7ce76</td>
      <td>408.0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>23.78</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_and_normal_double</td>
    </tr>
    <tr>
      <th>430</th>
      <td>66af9e9ba5224128a37f2b0db9f7c982</td>
      <td>576.0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>30.22</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_and_normal_double</td>
    </tr>
    <tr>
      <th>444</th>
      <td>68bab9dee0394863a2d15eeffa64d2fb</td>
      <td>588.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>18.66</td>
      <td>1</td>
      <td>0</td>
      <td>normal_double</td>
    </tr>
    <tr>
      <th>472</th>
      <td>6d9f997f587349d1b3977558dda3e2af</td>
      <td>576.0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>30.31</td>
      <td>1</td>
      <td>0</td>
      <td>mixed_and_normal_double</td>
    </tr>
  </tbody>
</table>
</div>


    
    === 최종 상세 데이터 ===
    


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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>amount</th>
      <th>case_type</th>
      <th>completed_row_cnt</th>
      <th>txn_cnt</th>
      <th>txn_amount_sum</th>
      <th>unique_offer_cnt</th>
      <th>duplicated_offer_id_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>432.0</td>
      <td>576.0</td>
      <td>22.05</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>22.05</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>22.05</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>22.05</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00ae03011f9f49b8a4b3e6d416678b0b</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_2</td>
      <td>504.0</td>
      <td>534.0</td>
      <td>618.0</td>
      <td>30.83</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>30.83</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00ae03011f9f49b8a4b3e6d416678b0b</td>
      <td>discount_7_3_7</td>
      <td>Discount_2</td>
      <td>576.0</td>
      <td>606.0</td>
      <td>618.0</td>
      <td>30.83</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>30.83</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00cf1bbec83f4a658f8994e556db4633</td>
      <td>discount_10_2_10</td>
      <td>Discount_2</td>
      <td>336.0</td>
      <td>438.0</td>
      <td>564.0</td>
      <td>33.28</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>33.28</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>00cf1bbec83f4a658f8994e556db4633</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>564.0</td>
      <td>33.28</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>33.28</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>00d7c95f793a4212af44e632fdc1e431</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>498.0</td>
      <td>504.0</td>
      <td>18.58</td>
      <td>mixed_instant_normal</td>
      <td>2</td>
      <td>1</td>
      <td>18.58</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>00d7c95f793a4212af44e632fdc1e431</td>
      <td>discount_10_2_7</td>
      <td>Discount_2</td>
      <td>504.0</td>
      <td>NaN</td>
      <td>504.0</td>
      <td>18.58</td>
      <td>mixed_instant_normal</td>
      <td>2</td>
      <td>1</td>
      <td>18.58</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>015c3d28c67e46aa95e9ec97c27220e8</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>510.0</td>
      <td>618.0</td>
      <td>22.12</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>22.12</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>015c3d28c67e46aa95e9ec97c27220e8</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>576.0</td>
      <td>582.0</td>
      <td>618.0</td>
      <td>22.12</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>22.12</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>01633b71b3a2457aa7d35d8bcc3afb5a</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>552.0</td>
      <td>600.0</td>
      <td>24.76</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>24.76</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>01633b71b3a2457aa7d35d8bcc3afb5a</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>594.0</td>
      <td>600.0</td>
      <td>24.76</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>24.76</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>01784d3e205548a594ba3fcdbdaaf17d</td>
      <td>bogo_10_10_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>444.0</td>
      <td>516.0</td>
      <td>24.67</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>24.67</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>01784d3e205548a594ba3fcdbdaaf17d</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>516.0</td>
      <td>24.67</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>24.67</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>01887dcd32b64feb807e2436548b6c87</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>408.0</td>
      <td>570.0</td>
      <td>25.04</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>25.04</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>01887dcd32b64feb807e2436548b6c87</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>522.0</td>
      <td>570.0</td>
      <td>25.04</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>25.04</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>01b6d7e8f0884deb936a8a7f15dba895</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>420.0</td>
      <td>558.0</td>
      <td>12.78</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>12.78</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>01b6d7e8f0884deb936a8a7f15dba895</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>510.0</td>
      <td>558.0</td>
      <td>12.78</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>12.78</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0215efe5136d4a038cb81eae92d59368</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>336.0</td>
      <td>342.0</td>
      <td>534.0</td>
      <td>11.96</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>11.96</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0215efe5136d4a038cb81eae92d59368</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>528.0</td>
      <td>534.0</td>
      <td>11.96</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>11.96</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>021adce38ab34ede96422ae107643fd5</td>
      <td>discount_10_2_7</td>
      <td>Discount_3</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>15.64</td>
      <td>mixed_instant_normal</td>
      <td>2</td>
      <td>1</td>
      <td>15.64</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>021adce38ab34ede96422ae107643fd5</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>576.0</td>
      <td>576.0</td>
      <td>15.64</td>
      <td>mixed_instant_normal</td>
      <td>2</td>
      <td>1</td>
      <td>15.64</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>021c1940868647efbcb40ccdb942813b</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>168.0</td>
      <td>192.0</td>
      <td>366.0</td>
      <td>15.49</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>15.49</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>021c1940868647efbcb40ccdb942813b</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>336.0</td>
      <td>348.0</td>
      <td>366.0</td>
      <td>15.49</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>15.49</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>021c1940868647efbcb40ccdb942813b</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_2</td>
      <td>504.0</td>
      <td>528.0</td>
      <td>624.0</td>
      <td>19.03</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>19.03</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>021c1940868647efbcb40ccdb942813b</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>576.0</td>
      <td>594.0</td>
      <td>624.0</td>
      <td>19.03</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>19.03</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>02a458e1233342b79caff81edbcc30a9</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>522.0</td>
      <td>618.0</td>
      <td>21.51</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>21.51</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>02a458e1233342b79caff81edbcc30a9</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>576.0</td>
      <td>588.0</td>
      <td>618.0</td>
      <td>21.51</td>
      <td>normal_double</td>
      <td>2</td>
      <td>1</td>
      <td>21.51</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>02abd909ebc94aca8766f3f0ee39db80</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>336.0</td>
      <td>396.0</td>
      <td>408.0</td>
      <td>14.80</td>
      <td>mixed_instant_normal</td>
      <td>2</td>
      <td>1</td>
      <td>14.80</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>02abd909ebc94aca8766f3f0ee39db80</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>414.0</td>
      <td>408.0</td>
      <td>14.80</td>
      <td>mixed_instant_normal</td>
      <td>2</td>
      <td>1</td>
      <td>14.80</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


# 위의 경우를 더블 카운팅으로 봐야할까..?
- 민수 생각 : 즉시 달성한 경우는 amount의 더블카운팅에 해당하는지 해당하지 않는지 명확히 알수있는 방법이 없다. 따라서, 308건의 즉시 달성 경우는 제거하고 보는 것이 확실한 데이터를 가지고 보는것이 아닐까 생각한다.

- 현규 생각: 총 amount를 구해야할 때만 308건의 데이터에 대한 처리를 해주는게 좋지 않을까?

### 추후 진행
- 1. 제거하자고 결론이 난 경우, 동일 person의 동일한 offer completed에 대해서, 다른 offer received만 있게 되고, 그 경우 amount는 최상위 행에만 남긴다면 총매출을 구하는데 오류가 없을 것이다.
- 2. 제거하지말자고 결론이 난 경우, 동일한 person이 동일한 시간에 completed -> received -> completed의 흐름을 동일한 amount로 만족했다고 가정하고, 해당 경우는 더블 카운팅에서 제외한다.

---


```python
print(f" viewed와 completed가 동일한 시간에 이루어진 경우 : {len(final_df[final_df['offer viewed'] == final_df['offer completed']])}건")
```

     viewed와 completed가 동일한 시간에 이루어진 경우 : 2719건
    


```python
# 1. '진성 전환(is_true_conversion)'이라는 새로운 기둥 만들기!
# 조건: viewed 시간이 존재하고, 그 시간이 completed 시간보다 작거나 같아야 함.
# viewed가 빈칸(NaN)인 건들은, 파이썬이 NaN <= 숫자를 비교할 때 자동으로 False(0)로 처리해 주기 때문에 에러 없이 걸러집니다
final_df['is_true_conversion'] = (final_df['offer viewed'] <= final_df['offer completed']).astype(int) # 동일 시간내에 viewed -> completed인지, completed-> viewed인지 알 수 있는 방법이 없음. viewed -> completed라고 가정해야함.

# 가슴 벅찬 최종 데이터 마트(Data Mart) 확인!!
display(final_df.head(10))
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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>amount</th>
      <th>is_true_conversion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>456.0</td>
      <td>414.0</td>
      <td>8.57</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>528.0</td>
      <td>14.11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>NaN</td>
      <td>576.0</td>
      <td>10.27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>168.0</td>
      <td>216.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_2</td>
      <td>576.0</td>
      <td>630.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>22.05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>432.0</td>
      <td>576.0</td>
      <td>22.05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>168.0</td>
      <td>186.0</td>
      <td>252.0</td>
      <td>11.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0020c2b971eb4e9188eac86d93036a77</td>
      <td>bogo_10_10_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>426.0</td>
      <td>510.0</td>
      <td>17.24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0020c2b971eb4e9188eac86d93036a77</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


## 위의 코드를 보면 동일 시간내에 viewed -> completed인지, completed-> viewed인지 알 수 있는 방법이 없음. viewed -> completed라고 가정해야함. 이것도 팀원간 논의 필요
- 1. 관대하게 보는경우 (<=)
- 2. 보수적을 보는경우 (<)


```python
# 프로모션(bogo, discount) 절차를 잘 적용하고 있는 고객 건수는 얼마나 되는가?
print(final_df["is_true_conversion"].sum())
```

    23267
    


```python
portfolio
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
      <th>reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>id</th>
      <th>web</th>
      <th>email</th>
      <th>mobile</th>
      <th>social</th>
      <th>offer_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>bogo</td>
      <td>ae264e3637204a6fb9bb56bc8210ddfd</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>bogo_10_10_7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>10</td>
      <td>5</td>
      <td>bogo</td>
      <td>4d5c57ea9a6940dd891ad53e9dbe8da0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>bogo_10_10_5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>informational</td>
      <td>3f207df678b143eea3cee63160fa8bed</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>informational_0_0_4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>bogo</td>
      <td>9b98b8c7a33c4b65b9aebfe6a799e6d9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>bogo_5_5_7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>20</td>
      <td>10</td>
      <td>discount</td>
      <td>0b1e1539f2cc45b7b9fa7c272da2e1d7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>discount_20_5_10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>discount</td>
      <td>2298d6c36e964ae4a3e7e9706d1fb8c2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>discount_7_3_7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>10</td>
      <td>10</td>
      <td>discount</td>
      <td>fafdcd668e3743c1bb461111dcafc2a4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>discount_10_2_10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>informational</td>
      <td>5a8bc65990b245e5a138643cd4eb9837</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>informational_0_0_3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>bogo</td>
      <td>f19421c1d4aa40978ebb69ca19b0e20d</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>bogo_5_5_5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>10</td>
      <td>7</td>
      <td>discount</td>
      <td>2906b810c7d4411798c6938adc9daaa5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>discount_10_2_7</td>
    </tr>
  </tbody>
</table>
</div>




```python
final_df
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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>amount</th>
      <th>is_true_conversion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>456.0</td>
      <td>414.0</td>
      <td>8.57</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>528.0</td>
      <td>14.11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>NaN</td>
      <td>576.0</td>
      <td>10.27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>168.0</td>
      <td>216.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_2</td>
      <td>576.0</td>
      <td>630.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>61037</th>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>60.0</td>
      <td>16.06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61038</th>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>336.0</td>
      <td>354.0</td>
      <td>384.0</td>
      <td>15.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61039</th>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>discount_10_2_7</td>
      <td>Discount_2</td>
      <td>408.0</td>
      <td>414.0</td>
      <td>414.0</td>
      <td>17.55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61040</th>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>discount_10_2_7</td>
      <td>Discount_3</td>
      <td>576.0</td>
      <td>582.0</td>
      <td>576.0</td>
      <td>14.23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>61041</th>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>168.0</td>
      <td>174.0</td>
      <td>198.0</td>
      <td>22.88</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>61042 rows × 8 columns</p>
</div>




```python
# profit(순수익) 구하기

# portfolio의 'offer_name'을 key, 'reward'를 value
offer_rewards = portfolio.set_index('offer_name')['reward'].to_dict()
final_df['reward_cost'] = final_df['offer_id'].map(offer_rewards)

# offer_id -> offer_name 이름 통일
final_df.rename(columns={'offer_id': 'offer_name'}, inplace=True)

# profit 계산하기! (amount - reward_cost)
final_df['profit'] = final_df['amount'] - final_df['reward_cost']

final_df = final_df[['person', 'offer_name', 'offer_cycle', 'offer received', 'offer viewed', 'offer completed', 'is_true_conversion', 'amount', 'reward_cost', 'profit']]
display(final_df.head(10))
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
      <th>person</th>
      <th>offer_name</th>
      <th>offer_cycle</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>offer completed</th>
      <th>is_true_conversion</th>
      <th>amount</th>
      <th>reward_cost</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>456.0</td>
      <td>414.0</td>
      <td>0</td>
      <td>8.57</td>
      <td>5</td>
      <td>3.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_10</td>
      <td>Discount_1</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>528.0</td>
      <td>0</td>
      <td>14.11</td>
      <td>2</td>
      <td>12.11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>discount_10_2_7</td>
      <td>Discount_1</td>
      <td>576.0</td>
      <td>NaN</td>
      <td>576.0</td>
      <td>0</td>
      <td>10.27</td>
      <td>2</td>
      <td>8.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_1</td>
      <td>168.0</td>
      <td>216.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00116118485d4dfda04fdbaba9a87b5c</td>
      <td>bogo_5_5_5</td>
      <td>Bogo_2</td>
      <td>576.0</td>
      <td>630.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>bogo_5_5_7</td>
      <td>Bogo_1</td>
      <td>504.0</td>
      <td>516.0</td>
      <td>576.0</td>
      <td>1</td>
      <td>22.05</td>
      <td>5</td>
      <td>17.05</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>discount_20_5_10</td>
      <td>Discount_1</td>
      <td>408.0</td>
      <td>432.0</td>
      <td>576.0</td>
      <td>1</td>
      <td>22.05</td>
      <td>5</td>
      <td>17.05</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0011e0d4e6b944f998e987f904e8c1e5</td>
      <td>discount_7_3_7</td>
      <td>Discount_1</td>
      <td>168.0</td>
      <td>186.0</td>
      <td>252.0</td>
      <td>1</td>
      <td>11.93</td>
      <td>3</td>
      <td>8.93</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0020c2b971eb4e9188eac86d93036a77</td>
      <td>bogo_10_10_5</td>
      <td>Bogo_1</td>
      <td>408.0</td>
      <td>426.0</td>
      <td>510.0</td>
      <td>1</td>
      <td>17.24</td>
      <td>10</td>
      <td>7.24</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0020c2b971eb4e9188eac86d93036a77</td>
      <td>bogo_10_10_7</td>
      <td>Bogo_1</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>10</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 완벽한 정상 흐름(received -> viewed -> completed) 필터링
# 조건: 받은 시간 <= 본 시간 <= 달성한 시간
true_funnel_mask = (final_df['offer received'] <= final_df['offer viewed']) & (final_df['offer viewed'] <= final_df['offer completed']) # 이것도 아까처럼 (<= / <) 신경써야함!
best_cases_df = final_df[true_funnel_mask]

# offer_name별 지표 만들기
performance_df = best_cases_df.groupby('offer_name').agg(
    conversion_count=('person', 'count'),      # 이상적인 전환 카운트
    total_revenue=('amount', 'sum'),           # 총 발생 매출
    total_profit=('profit', 'sum'),            # 총 순수익
    avg_profit=('profit', 'mean')              # 건당 평균 순수익 
).reset_index()

performance_df = performance_df.sort_values(by='total_profit', ascending=False)

display(performance_df)
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
      <th>offer_name</th>
      <th>conversion_count</th>
      <th>total_revenue</th>
      <th>total_profit</th>
      <th>avg_profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>discount_10_2_10</td>
      <td>4576</td>
      <td>82824.93</td>
      <td>73672.93</td>
      <td>16.099854</td>
    </tr>
    <tr>
      <th>7</th>
      <td>discount_7_3_7</td>
      <td>4348</td>
      <td>74016.92</td>
      <td>60972.92</td>
      <td>14.023211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bogo_5_5_5</td>
      <td>3514</td>
      <td>70093.72</td>
      <td>52523.72</td>
      <td>14.946989</td>
    </tr>
    <tr>
      <th>5</th>
      <td>discount_10_2_7</td>
      <td>2102</td>
      <td>42263.73</td>
      <td>38059.73</td>
      <td>18.106437</td>
    </tr>
    <tr>
      <th>0</th>
      <td>bogo_10_10_5</td>
      <td>2739</td>
      <td>65289.48</td>
      <td>37899.48</td>
      <td>13.836977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bogo_10_10_7</td>
      <td>2582</td>
      <td>61508.30</td>
      <td>35688.30</td>
      <td>13.821960</td>
    </tr>
    <tr>
      <th>6</th>
      <td>discount_20_5_10</td>
      <td>1300</td>
      <td>34579.64</td>
      <td>28079.64</td>
      <td>21.599723</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bogo_5_5_7</td>
      <td>2106</td>
      <td>37422.05</td>
      <td>26892.05</td>
      <td>12.769255</td>
    </tr>
  </tbody>
</table>
</div>



```python
# # 'offer_name' (쿠폰 종류) 별로 묶어서 성과 분석!
# promotion_performance = final_df.groupby('offer_name').agg(
#     total_sent=('person', 'count'),                        # 몇 명에게 뿌렸나?
#     true_conversion_rate=('is_true_conversion', 'mean'),       # 진성 전환율(%)
#     avg_amount_per_conv=('amount', 'mean'),                # 전환된 건당 평균 결제액
#     avg_profit_per_conv=('profit', 'mean')                 # 전환된 건당 평균 순수익(마진)!
# ).reset_index()

# # 보기 좋게 %로 변환
# promotion_performance['true_conversion_rate'] = (promotion_performance['true_conversion_rate'] * 100).round(1).astype(str) + '%'
# # 소수점 2자리로 깔끔하게 정리
# promotion_performance['avg_amount_per_conv'] = promotion_performance['avg_amount_per_conv'].round(2)
# promotion_performance['avg_profit_per_conv'] = promotion_performance['avg_profit_per_conv'].round(2)

# # 수익성(avg_profit)이 높은 순서대로 정렬!
# promotion_performance = promotion_performance.sort_values('avg_profit_per_conv', ascending=False)

# display(promotion_performance)
```


```python
# 총 매출액은 transactions_df -> 중복되지 않음/여기서 구해야함
# true_conversion_rate (진성 전환율)
# avg_amount_per_conv (전환당 평균 결제액 - 객단가)
# avg_profit_per_conv (전환당 평균 순수익)



```

## 스타벅스 프로모션 성과 측정을 위한 데이터 마트 구축 보고

1. 오프닝: 우리의 목표 (Hook)

"저는 스타벅스 앱 내에 무작위로 쌓여있던 수십만 건의 이벤트 로그를, 우리 프로모션의 진짜 ROI(투자 대비 수익률)를 측정할 수 있는 '분석용 데이터 마트'로 탈바꿈시킨 결과를 공유하고자 합니다."

2. AS-IS: 기존 데이터의 문제점 (Pain Point)

"기존 데이터는 세로로 길게 쌓이는 단순 로그(Log) 형태였고, 핵심 정보인 결제액과 보상액은 value라는 딕셔너리 안에 숨겨져 있었습니다.
이 상태에서는 '어떤 쿠폰이 매출을 얼마나 견인했는지' 연결할 수가 없었습니다. 단순히 '어제 매출 얼마야?'는 알 수 있어도, '이 BOGO 쿠폰 때문에 발생한 매출이 얼마야?'라는 질문에는 대답할 수 없는 맹인이었던 셈이죠."

3. TO-BE: 고객 여정의 평면화 (Solution)

"그래서 저는 이 데이터를 '고객 1명 + 쿠폰 1장' 단위로 가로로 넓게 펼치는 '평면화(Flattening)' 작업을 진행했습니다. 쿠폰을 받고(Received), 보고(Viewed), 결제하는(Completed) 모든 과정을 한 줄의 타임라인으로 만들고, 거기에 영수증 데이터(Amount)를 1:1로 결합했습니다.
왜 이렇게 했을까요? 바로 핵심 KPI를 정확하게 측정하기 위해서입니다."

4. 🎯 우리가 얻게 된 3가지 핵심 KPI (Highlight)

"이 데이터 마트 구축을 통해, 우리는 이제 껍데기뿐인 '단순 달성률'을 버리고 다음과 같은 진짜 비즈니스 지표를 뽑아낼 수 있게 되었습니다."

KPI 1. 진성 전환율 (True Conversion Rate)

의미: "쿠폰을 달성한 전체 고객이 아니라, '광고를 본 뒤에 결제한(Viewed <= Completed)' 진짜 마케팅 성과만 발라냈습니다. 원래 커피를 마시려다 운 좋게 쿠폰을 쓴 '체리피커'의 허수를 제거한 순도 100%의 전환율입니다."

KPI 2. 프로모션별 평균 순수익 (Average Profit per Offer)

의미: "결제액(Amount)에서 우리가 퍼준 혜택(Reward Cost)을 뺀 **진짜 마진(Profit)**을 쿠폰별로 계산했습니다. 이제 '매출은 높은데 적자 나는 쿠폰'과 '전환율은 낮아도 흑자 폭이 큰 알짜 쿠폰'을 완벽하게 구분할 수 있습니다."

KPI 3. 객단가 방어율 (Average Order Value, AOV)

의미: "할인 폭이 큰 쿠폰일수록 고객이 딱 조건만 채우고 이탈하는지, 아니면 추가 결제를 일으켜 객단가(Amount)를 높이는지 비교할 수 있는 토대를 마련했습니다."

5. 🚨 주의사항: 기여도(Attribution)와 더블 카운팅 (Risk Management)

"한 가지 팀원분들께 주의를 당부드리고 싶은 점이 있습니다
고객이 1번 결제해서 BOGO와 할인 쿠폰 2개를 동시에 달성한 경우, 우리는 두 쿠폰 모두에 매출 기여도를 100% 인정했습니다.
따라서 이 데이터는 '어떤 프로모션이 더 강력한가'를 비교(기여도 분석)할 때만 사용해야 하며, 이 표의 수익을 다 더해서 '전사 총매출'로 보고하는 더블 카운팅의 오류를 범해서는 안 됩니다. 전사 총매출은 원본 영수증(Transaction) 데이터로 별도 산출해야 합니다."

6. Next Step 및 마무리

"현재 BOGO와 Discount 쿠폰에 대한 분석 파이프라인은 완성이 되었습니다. 다음 스텝으로는, 달성(Completed) 로그가 아예 남지 않는 '정보성(Informational) 광고'가 고객의 결제에 미친 영향을 '타임 윈도우(Time-window)' 방식으로 추적하는 로직을 추가로 개발할 예정입니다.

---

## informational의 마케팅효과를 어떻게 확인할까?
### informational의 duration 내에 고객이 결제를 한 정보가 있다면, informational의 마케팅효과가 있다고 하자.


```python
all_merge_df
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
      <th>amount</th>
      <th>offer_id</th>
      <th>transcript_reward</th>
      <th>gender</th>
      <th>age</th>
      <th>became_member_on</th>
      <th>income</th>
      <th>portfolio_reward</th>
      <th>difficulty</th>
      <th>duration</th>
      <th>offer_type</th>
      <th>web</th>
      <th>email</th>
      <th>mobile</th>
      <th>social</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55972</th>
      <td>55972</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>168</td>
      <td>NaN</td>
      <td>informational_0_0_3</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>informational</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>77705</th>
      <td>77705</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>192</td>
      <td>NaN</td>
      <td>informational_0_0_3</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>informational</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>89291</th>
      <td>89291</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>transaction</td>
      <td>228</td>
      <td>22.16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>113605</th>
      <td>113605</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer received</td>
      <td>336</td>
      <td>NaN</td>
      <td>informational_0_0_4</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>139992</th>
      <td>139992</td>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>offer viewed</td>
      <td>372</td>
      <td>NaN</td>
      <td>informational_0_0_4</td>
      <td>NaN</td>
      <td>M</td>
      <td>33.0</td>
      <td>2017-04-21</td>
      <td>72000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>informational</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>258361</th>
      <td>258361</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>transaction</td>
      <td>576</td>
      <td>14.23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>258362</th>
      <td>258362</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>offer completed</td>
      <td>576</td>
      <td>NaN</td>
      <td>discount_10_2_7</td>
      <td>2.0</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>discount</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>262475</th>
      <td>262475</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>offer viewed</td>
      <td>582</td>
      <td>NaN</td>
      <td>discount_10_2_7</td>
      <td>NaN</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>discount</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>274809</th>
      <td>274809</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>transaction</td>
      <td>606</td>
      <td>10.12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>289924</th>
      <td>289924</td>
      <td>ffff82501cea40309d5fdd7edcca4a07</td>
      <td>transaction</td>
      <td>648</td>
      <td>18.91</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>45.0</td>
      <td>2016-11-25</td>
      <td>62000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>306534 rows × 19 columns</p>
</div>




```python
# informational만 뽑기
info_df = all_merge_df[all_merge_df['offer_type'] == 'informational'].copy()
# offer_cycle 만들기
info_df = info_df.sort_values(['person', 'offer_id', 'time'])
info_df['is_received'] = (info_df['event'] == 'offer received').astype(int)
info_df['offer_cycle'] = info_df.groupby(['person', 'offer_id'])['is_received'].cumsum()

# 피벗 테이블
info_pivot = info_df.pivot_table(
    index=['person', 'offer_id', 'offer_cycle', 'duration'], # duration 추가
    columns='event', values='time', aggfunc='min'
).reset_index()

# expire_time : 유효기간 /// duration은 일(days)단위, 시간(hours)과 맞추기
info_pivot['expire_time'] = info_pivot['offer received'] + (info_pivot['duration'] * 24)

# 결제 정보 가져오기
all_transactions = all_merge_df[all_merge_df['event'] == 'transaction'][['person', 'time', 'amount']]

# 모든 transaction을 person을 기준으로 left join
info_trans_merge = pd.merge(info_pivot, all_transactions, on='person', how='left')

print(f"left join 후 행 크기 : {len(info_trans_merge)}")
# 필터링: 유효기간 내에 결제한 것만 추리기
success_mask = (
    info_trans_merge['offer viewed'].notna() &
    (info_trans_merge['time'] >= info_trans_merge['offer viewed']) &
    (info_trans_merge['time'] <= info_trans_merge['expire_time'])
)
valid_info_transactions = info_trans_merge[success_mask].copy()
print(f"유효기간 내 결제 건 수 : {len(valid_info_transactions)}")

# 더블 카운팅이 생길 수 있음. 해결하기 위한 코드 [일단 'Last Touch'라는 방식을 사용] # First Touch / Last Touch / Multi-Touch
# 팀원들과 회의 필요
# 결제 시간(time)과 알림 본 시간(offer viewed)의 차이 계산
valid_info_transactions['time_diff'] = valid_info_transactions['time'] - valid_info_transactions['offer viewed']

# 사람과 결제시간을 묶어 정렬하되, time_diff가 가장 작은(가장 최근) 순서로 정렬
valid_info_transactions = valid_info_transactions.sort_values(by=['person', 'time', 'time_diff'])

# 동일한 (사람+결제시간)이 여러 개라면, 맨 위(가장 최근 알림) 1개만 남기고 중복 제거
valid_info_transactions = valid_info_transactions.drop_duplicates(subset=['person', 'time'], keep='first')
print(f"Last Touch 방식으로 더블카운팅 문제 해결 후 건 수: {len(valid_info_transactions)}")
display(valid_info_transactions)
```

    left join 후 행 크기 : 121643
    유효기간 내 결제 건 수 : 10612
    Last Touch 방식으로 더블카운팅 문제 해결 후 건 수: 10564
    


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
      <th>person</th>
      <th>offer_id</th>
      <th>offer_cycle</th>
      <th>duration</th>
      <th>offer received</th>
      <th>offer viewed</th>
      <th>expire_time</th>
      <th>time</th>
      <th>amount</th>
      <th>time_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>informational_0_0_3</td>
      <td>1</td>
      <td>3.0</td>
      <td>168.0</td>
      <td>192.0</td>
      <td>240.0</td>
      <td>228.0</td>
      <td>22.16</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0009655768c64bdeb2e877511632db8f</td>
      <td>informational_0_0_4</td>
      <td>1</td>
      <td>4.0</td>
      <td>336.0</td>
      <td>372.0</td>
      <td>432.0</td>
      <td>414.0</td>
      <td>8.57</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0020ccbbb6d84e358d3414a3ff76cffd</td>
      <td>informational_0_0_3</td>
      <td>1</td>
      <td>3.0</td>
      <td>408.0</td>
      <td>408.0</td>
      <td>480.0</td>
      <td>426.0</td>
      <td>8.93</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0020ccbbb6d84e358d3414a3ff76cffd</td>
      <td>informational_0_0_3</td>
      <td>1</td>
      <td>3.0</td>
      <td>408.0</td>
      <td>408.0</td>
      <td>480.0</td>
      <td>432.0</td>
      <td>20.08</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0020ccbbb6d84e358d3414a3ff76cffd</td>
      <td>informational_0_0_3</td>
      <td>1</td>
      <td>3.0</td>
      <td>408.0</td>
      <td>408.0</td>
      <td>480.0</td>
      <td>450.0</td>
      <td>10.76</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>121583</th>
      <td>ffeaa02452ef451082a0361c3ca62ef5</td>
      <td>informational_0_0_3</td>
      <td>2</td>
      <td>3.0</td>
      <td>408.0</td>
      <td>420.0</td>
      <td>480.0</td>
      <td>420.0</td>
      <td>19.99</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>121603</th>
      <td>fff0f0aac6c547b9b263080f09a5586a</td>
      <td>informational_0_0_4</td>
      <td>2</td>
      <td>4.0</td>
      <td>576.0</td>
      <td>636.0</td>
      <td>672.0</td>
      <td>660.0</td>
      <td>17.10</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>121624</th>
      <td>fff3ba4757bd42088c044ca26d73817a</td>
      <td>informational_0_0_3</td>
      <td>2</td>
      <td>3.0</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>576.0</td>
      <td>540.0</td>
      <td>388.22</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>121625</th>
      <td>fff3ba4757bd42088c044ca26d73817a</td>
      <td>informational_0_0_3</td>
      <td>2</td>
      <td>3.0</td>
      <td>504.0</td>
      <td>540.0</td>
      <td>576.0</td>
      <td>552.0</td>
      <td>26.09</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>121635</th>
      <td>fffad4f4828548d1b5583907f2e9906b</td>
      <td>informational_0_0_3</td>
      <td>1</td>
      <td>3.0</td>
      <td>168.0</td>
      <td>168.0</td>
      <td>240.0</td>
      <td>198.0</td>
      <td>5.63</td>
      <td>30.0</td>
    </tr>
  </tbody>
</table>
<p>10564 rows × 10 columns</p>
</div>


- offer_cycle이 헷갈릴수 있는데 40,41,42를보면 동일한 person, offer_id인데 offer_cycle이 1임을 볼 수있다. 1,2,3으로 늘어나야하는거 아닌가?
- No, 이유는, offer received를 보면 동일한 시간임. 즉, 동일한 시간에 받은 쿠폰이 결제에 영향을 3번 미침(time)을 알 수있다.



```python
# 건당 얼마 썼는지 계산
instance_revenue = valid_info_transactions.groupby(['person', 'offer_id', 'offer_cycle']).agg(
    total_amount=('amount', 'sum')
).reset_index()

# 최종 지표
info_performance = instance_revenue.groupby('offer_id').agg(
    conversion_count=('person', 'count'), # 마케팅 성공 횟수(전환 횟수)
    total_revenue=('total_amount', 'sum')  # 총 발생 매출 (= 순수익)
).reset_index()

info_performance = info_performance.sort_values(by='total_revenue', ascending=False)

display(info_performance)
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
      <th>offer_id</th>
      <th>conversion_count</th>
      <th>total_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>informational_0_0_3</td>
      <td>3646</td>
      <td>73772.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>informational_0_0_4</td>
      <td>2295</td>
      <td>59222.84</td>
    </tr>
  </tbody>
</table>
</div>


### 인사이트? 나의생각 : 하지만, bogo, discount와 달리 informational은 성격이 다르기 때문에 두 케이스를 같은 선상에 놓고 비교하기에는 무리가 있다.

### 내일 논의할 사항
- 1. 즉시 달성의 경우, 해석이 어려우니 해당 행들을 제거할지 안할지
- 2. 코드에서의 부등호 (<= / <)
- 3. informational의 효과에 대한 정의
- 4. informational에서 더블카운팅 해결 방식(First Touch / Last Touch / Multi-Touch)


```python
!jupyter nbconvert --to markdown "03_eda_v2.ipynb"
```

    [NbConvertApp] Converting notebook 03_eda_v2.ipynb to markdown
    [NbConvertApp] Writing 82709 bytes to 03_eda_v2.md
    
