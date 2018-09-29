# Trading Agents for Quantitative trading

## Intro

There are trading agents that learns to buy and sell stocks by connecting with trading gym. It consists of 4 agents having each own's role.

- Buy Signal Agent (BSA), which generates a buy signal.
- Buy Order Agent (BOA), which receives a buy signal and makes a purchase at the lowest price within a certain period of time
- Sell Signal Agent (SSA) which generates sell signal with receiving buyer price and quantity, market information
- Sell Order Agent (SOA) which receives sell signal and sells at highest price within a certain time. 

Each Agents have an independent reward, but some parts of the rewards are shared with other Agents and eventually move to a single goal of 'Trading Return'.



## Architecture
![](/materials/architecture.png)



## How to install
You can clone two repository into your local computer or cloud whatever. And you can run agent after including traidng gym as submodule or add gym's root into Python Path and then you can run trading agent itself. so simple!
 First, You have to install Trading Gym. This is [link](https://github.com/6-Billionaires/trading-gym).



## Quick Run

```
import sys
from gym_core import tgym

class ENV(tgym.tradingGymEnv):
	def _rewards(self, observation, action, done, info):
		# calcuate reward
		return reward
```



## Buy Signal Agent (BSA)

- GOAL 

   The Buy Signal Agent (BSA) continues to monitor the market and finds the stock price likely to rise more than 2% in two minutes. If a particular stock is expected to rise more than 2% in two minutes, send a buy signal to the Buy Order Agent. Also, BSA receives rewards based on how other agents did well.



- REWARD

   The area occupied by the price of 2 minutes compared to the price at the time of generating the buy signal (-1% to 1% clipping with the upper limit at 2% for 120 seconds and the lower limit at -2%
   Also, get rewards based on how toher agents did well. BSA gets other agents' reward mutipled with specific number(K).




- STATE

1. Orderbook data during N seconds.

   It have 41 columns with time, bid prices (10), bid volumes (10), ask prices (10), ask volumes (10)

2. Executed trade data during N seconds.

   It have 12 columns with time, price(last executed), Buy/Sell/Total executed, Buy/Sell/Total Weighted Price, Execution Strength, Open/High/Low Price



## Buy Order Agent (BOA)

- GOAL 

   When the Buy Order Agent (BOA) receives a buy signal from BSA, it proceeds to buy at the lowest possible price within two minutes. If BOA does not make a purchase within the time limit, BOA will be forced to buy at now . The BOA receives a reward based on the price how much BOA bought the stock cheaper than the price when BSA has given the buy signal.



- REWARD
   How much cheaper than the price when you received a signal from BSA (-2% upper limit, -1 ~ 1% Clipping with 2% lower limit)
    Also, get rewards based on how other agents did well(except, BSA). BOA gets other agents' reward mutipled with specific number(K).



- STATE
  Same with other agents.



## Sell Signal Agent (SSA)

- GOAL 

  The Sell Signal Agent (SSA) observes the market from the time BOA purchases the stock, and searches for a time when the share price is likely to fall by more than 2% within the remaining time. If you believe a particular stock will fall by more than 2%, send Sell Signal to the Sell Order Agent. If the SSA signal does not occur within 2 minutes of the BSA signal, the signal is forced to make signal. 



- REWARD

  The area occupied by the price for 2 minutes compared to the price at the time of generating the Sell Signal (the upper limit of the area at -2% for 120 seconds and the lower limit of the area at 2% for -1 to 1 clipping)



- STATE
 Same with other agents.



## Sell Order Agent (SOA)

- GOAL 

   The Sell Order Agent (SOA) sells at the most expensive price within the remaining time when receiving the Sell Signal from the SSA. If you do not sell within 2 minutes of the BSA signaling, you will immediately sell it to the market. The buyer receives a reward based on the price bought by the BOA against the stock price at the time of receiving the Sell Signal from the SSA.



- REWARD

  The selling price (2% upper limit, -1 ~ 1 clipping with the lower limit of -2%) when the SSA receives a signal.


- STATE
  Same with other agents.
