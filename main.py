from dqn_agent import DQNAgent
from trading_gym.core import tgym


if __name__ == '__main__':
    EPISODES = 10000
    RENDER = False
    ACTION_SIZE = 2

    env = tgym.TradingGymEnv(episode_type='0')
    state = env.init_observation()  # to calculate length of state data
    agent = DQNAgent.DQNAgent(state_size=len(state), action_size=ACTION_SIZE)

    for ep in range(EPISODES):
        score = 0
        done = False
        env.reset()
        state = env.init_observation()

        while not done:
            if RENDER:
                env.render()

            action = agent.get_action(state)
            next_state, _, done, info = env.step(action)
            reward = agent.calc_reward(info)

            agent.append_sample(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state
            score += reward

            if done:
                agent.update_target_model()
                agent.save_model()
                print('profit :', score, 'memory :', len(agent.memory), 'epsilon :', round(agent.epsilon, 5))

