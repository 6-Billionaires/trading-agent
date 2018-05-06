import core.tagent as tagent
import core.tenv as MarketEnv

class RandomAgent(tagent) :
    pass


if __name__ == '__main__' :
    """
    In here, we provide a very simple sample to help you to write your own agent based one this. 

    Create a market environment, instantiate a random agent, and run the agent for one episode.
    """
    # Samsung Electronics ticker = 005930 , 1-second bars by default
    env = MarketEnv("005930", interval="1s", episode_steps=100)
    agent = RandomAgent(env.action_space)       # Actions are continuous from -1 = go short to +1 = go long.  0 is go flat.  Sets absolute target position.
    observation = env.reset()       # An observation is a numpy float array, values: time, bid, bidsize, ask, asksize, last, lastsize, lasttime, open, high, low, close, vwap, volume, open_interest, position, unrealized_gain
    done = False
    total_reward = 0.0              # Reward is the profit realized when a trade closes
    while not done:
        env.render()
        observation, reward, done, info = env.step(agent.act(observation))
        total_reward += reward

    print('\nTotal profit: {:.2f}'.format(total_reward))        # Sairen will automatically (try to) cancel open orders and close positions on exit
