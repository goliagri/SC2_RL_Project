import gym
import time

#env = gym.make('ma_gym:Lumberjacks-v0')
env = gym.make('ma_gym:Checkers-v3')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    print(obs_n)
    print(env.action_space)
    time.sleep(.1)
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
env.close()