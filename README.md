# Trading Agents for Quantitative trading

## Intro

Trading Gym 과 연결하여 주식 단타 매매를 학습하는 Trading Agent 이다. 총 4개의 Agent를 사용하여 학습하는데, 넓은 간격의 시장 데이터를 관찰하면서 매수 시그널을 생성하는 Buy Signal Agent (BSA), 매수 시그널을 받아 일정 시간 안에 최저가로 매수를 하는 Buy Order Agent (BOA), 주식 매수가격과 수량, 시장 정보를 받아 매도 시그널을 생성하는 Sell Signal Agent (SSA), 매도 시그널을 받아 일정 시간 안에 최고가로 매도하는 Sell Order Agent (SOA) 로 구성된다. 각각의 Agent는 독립적인 Reward 를 갖고 있지만, Reward의 일정 부분은 다른 Agent들과 공유되어 최종적으로는 '수익률' 이라는 하나의 목표로 움직이게 된다.



## Architecture
![](/materials/architecture.png)



## How to run
1. 로컬 컴퓨터에 Trading-Gym 과 Trading-Agent 를 clone 받는다.
2. 학습하고자 하는 Agent 에서 Gym을 임포트하고 gym_core.tgym.TradingGymEnv를 상속 받아 새로운 ENV를 생성한다.
3. ENV에서 _rewards() 함수를 작성한다. observation은 전처리된 시장 관찰 결과를, action은 Agent의 행동을, info는 결과값을 담고 있다. 이를 토대로 reward 값을 계산하여 return 해 주면 된다.
4. Agent 를 구현한다. (keras-rl 문서 참조)




#### ENV 예시

import sys

sys.path.insert(0, 'trading-gym 경로')

from gym_core import tgym

class ENV(tgym.tradingGymEnv):

​	def _rewards(self, observation, action, done, info):

​		# calcuate reward

​		return reward




## Buy Signal Agent (BSA)

GOAL 

전처리된 장기 데이터(초단타 기준=약 1~2분) 만을 State로 입력 받아 일정 시간(default=60s) 안에 일정 비율(default=2%) 만큼 상승하는 상황(이하 Upward TP)을 찾는 것을 목표로 한다. 



REWARD

1. 일정 시간 안에 일정 비율만큼 올랐을 경우 reward로 1을 받고, 일정 시간 안에 일정 비율(default=-2%) 만큼 떨어졌을 경우 reward를 -1로 받는다. 두 조건 다 만족했을 때는 먼저 만족한 조건을 기준으로 reward를 받고, 둘 다 만족하지 못했을 때는 -1 ~ 1 로 clipping 된 reward를 (마지막 가격 / 처음 가격) ratio 기준으로 받게 된다.
2. BOA reward * 0.1
3. SOA reward * 0.1

위 리워드를 전부 합산한다.



STATE

1. N초간의 호가창 정보 (N=60초)
   1.1 N초간의 호가창 정보를 어제 종가로 scailing 한것 ( ex 60초 * 40columns(매수호가, 매수잔량, 매도호가, 매도잔량))
    -> 60 * 40  = 2400
2. N초간의 체결량 정보
   2.1 N초간의 체결량 정보를 scailing 한것 ( ex 60초 * 7 columns)
    -> 60 * 7
3. Market Info
   3.1 N초 동안의 그 종목의 OPEN, HIGH, LOW
    3.2 N초 동안의 코스피, 코스닥(지수) 흐름



NETWORK

1. RRL
2. Temporal CNN



## Buy Order Agent (BOA)

GOAL

BSA로부터 signal을 받았을 때, 일정 시간(default=60s) 안에 signal을 받았을 때의 가격 대비 가장 싼 가격으로 주식을 매수하는 것을 목표로 한다.



REWARD

1. Signal 을 받았을 때 가격 대비 얼마나 싸게 매수했는가를 -1 ~ 1 로 clipping 한 값
2. SOA reward * 0.1

위 리워드를 전부 합산한다.



STATE

1. N초간의 호가창 정보 (N=60초)
   1.1 N초간의 호가창 정보를 어제 종가로 scailing 한것 ( ex 60초 * 40columns(매수호가, 매수잔량, 매도호가, 매도잔량))
    -> 60 * 40  = 2400
2. N초간의 체결량 정보
   2.1 N초간의 체결량 정보를 scailing 한것 ( ex 60초 * 7 columns)
    -> 60 * 7
3. Market Info
   3.1 N초 동안의 그 종목의 OPEN, HIGH, LOW
    3.2 N초 동안의 코스피, 코스닥(지수) 흐름



## Sell Signal Agent (SSA)

추가예정



## Sell Order Agent (SOA)

추가예정