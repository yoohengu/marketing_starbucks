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
portfolio = pd.read_csv('../data/portfolio.csv')
profile = pd.read_csv('../data/profile.csv')
transcript = pd.read_csv('../data/transcript.csv')
```

---


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


```python
# profile의 필요없는 Unnamed:0 컬럼 제거
profile = profile.drop('Unnamed: 0', axis=1)

# transcript 기준으로 profile 데이터를 Left Join
merged_df = pd.merge(transcript, profile, left_on='person', right_on='id', how='left')

# 필요 없는 id 컬럼(person과 중복)은 버리기
merged_df = merged_df.drop(columns='id')
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
# 1. 시간 순서대로 예쁘게 줄 세우기 (시간이 꼬이면 안 되니까 필수)
offers_df = offers_df.sort_values(['person', 'offer_id', 'time'])

# 2. 'received' 이벤트가 등장할 때마다 1, 아니면 0인 깃발(Flag) 만들기
offers_df['is_received'] = (offers_df['event'] == 'offer received').astype(int)

# 3. [마법의 함수] 사람과 쿠폰 단위로 묶어서, 깃발을 누적해서 더하기 (Cumsum)
offers_df['offer_cycle'] = offers_df.groupby(['person', 'offer_id'])['is_received'].cumsum()

# # 4. 이제 찝찝함 없이 피벗 돌리기 (기준점에 offer_cycle 추가!)
# pivot_df = offers_df.pivot_table(
#     index=['person', 'offer_id', 'offer_cycle'],  # "누구의 / 어떤 쿠폰의 / 몇 회차인가?"
#     columns='event',
#     values='time',
#     aggfunc='min'  # 이제 한 회차 안에는 중복이 없으니 min을 써도 아무 왜곡이 안 일어납니다
# ).reset_index()
pivot_df = offers_df.pivot_table(
    index=['person', 'offer_id', 'offer_cycle'],
    columns='event',
    values={'time': ['offer received', 'offer viewed', 'offer completed'],
            'transcript_reward': 'offer completed'},  # completed 시점의 reward만
    aggfunc='min'
).reset_index()

# 깔끔하게 정리
pivot_df.columns.name = None
pivot_df = pivot_df[['person', 'offer_id', 'offer_cycle', 'offer received', 'offer viewed', 'offer completed']]

display(pivot_df.head())
display(pivot_df.shape)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[42], line 27
         25 # 깔끔하게 정리
         26 pivot_df.columns.name = None
    ---> 27 pivot_df = pivot_df[['person', 'offer_id', 'offer_cycle', 'offer received', 'offer viewed', 'offer completed']]
         29 display(pivot_df.head())
         30 display(pivot_df.shape)
    

    File c:\Users\hkhk3\Desktop\sparta\project\tp3\marketing_starbucks\.venv\Lib\site-packages\pandas\core\frame.py:4384, in DataFrame.__getitem__(self, key)
       4382     if is_iterator(key):
       4383         key = list(key)
    -> 4384     indexer = self.columns._get_indexer_strict(key, "columns")[1]
       4386 # take() does not accept boolean indexers
       4387 if getattr(indexer, "dtype", None) == bool:
    

    File c:\Users\hkhk3\Desktop\sparta\project\tp3\marketing_starbucks\.venv\Lib\site-packages\pandas\core\indexes\multi.py:3239, in MultiIndex._get_indexer_strict(self, key, axis_name)
       3236 if len(keyarr) and not isinstance(keyarr[0], tuple):
       3237     indexer = self._get_indexer_level_0(keyarr)
    -> 3239     self._raise_if_missing(key, indexer, axis_name)
       3240     return self[indexer], indexer
       3242 return super()._get_indexer_strict(key, axis_name)
    

    File c:\Users\hkhk3\Desktop\sparta\project\tp3\marketing_starbucks\.venv\Lib\site-packages\pandas\core\indexes\multi.py:3257, in MultiIndex._raise_if_missing(self, key, indexer, axis_name)
       3255 cmask = check == -1
       3256 if cmask.any():
    -> 3257     raise KeyError(f"{keyarr[cmask]} not in index")
       3258 # We get here when levels still contain values which are not
       3259 # actually in Index anymore
       3260 raise KeyError(f"{keyarr} not in index")
    

    KeyError: "['offer received' 'offer viewed' 'offer completed'] not in index"



```python
# 1. 원본에서 offer_id와 offer_type 짝꿍 사전 만들기
offer_dict = offers_df[['offer_id', 'offer_type']].drop_duplicates().set_index('offer_id')['offer_type'].to_dict()

# 2. 피벗 테이블의 offer_id를 보고, 임시로 쿠폰 타입(bogo, discount)을 가져오기
temp_offer_type = pivot_df['offer_id'].map(offer_dict)

# 3. [핵심] 기존 숫자였던 'offer_cycle' 컬럼 위에 곧바로 덮어쓰기! 
pivot_df['offer_cycle'] = temp_offer_type.str.capitalize() + '_' + pivot_df['offer_cycle'].astype(str)

```


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

reward_df = offers_df[offers_df['event'] == 'offer completed'][['person', 'offer_id', 'offer_cycle', 'transcript_reward']]
final_df = final_df.merge(
    reward_df,
    on=['person', 'offer_id', 'offer_cycle'],
    how='left'
)

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


```python
amount_by_offer = (
    final_df[final_df['amount'].notna()]
    .groupby('offer_id')['amount']
    .sum()
    .reset_index()
    .rename(columns={'amount': 'total_amount'})
    .sort_values('total_amount', ascending=False)
)

# 총합 행 추가
total_row = pd.DataFrame([{'offer_id': '합계', 'total_amount': amount_by_offer['total_amount'].sum()}])
amount_by_offer = pd.concat([amount_by_offer, total_row], ignore_index=True)

display(amount_by_offer)
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
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>discount_10_2_10</td>
      <td>96749.46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>discount_7_3_7</td>
      <td>89794.47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bogo_10_10_7</td>
      <td>87109.94</td>
    </tr>
    <tr>
      <th>3</th>
      <td>discount_20_5_10</td>
      <td>85660.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bogo_5_5_5</td>
      <td>83146.93</td>
    </tr>
    <tr>
      <th>5</th>
      <td>discount_10_2_7</td>
      <td>81631.79</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bogo_10_10_5</td>
      <td>78562.14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bogo_5_5_7</td>
      <td>77066.31</td>
    </tr>
    <tr>
      <th>8</th>
      <td>합계</td>
      <td>679721.20</td>
    </tr>
  </tbody>
</table>
</div>


---


```python
!jupyter nbconvert --to markdown "04_event.ipynb"
```

    [NbConvertApp] Converting notebook 04_event.ipynb to markdown
    [NbConvertApp] Writing 6331 bytes to 04_event.md
    
