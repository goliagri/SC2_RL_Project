import numpy as np
import random
import keras.api._v2.keras as keras
from IPython.display import clear_output
from collections import deque
import gym
from keras import layers
from keras import Model, Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam
from tqdm import tqdm

env = gym.make('ma_gym:Checkers-v3')
env.reset()
env.render()

NUM_AGENTS = len(env.observation_space.sample())
print('Number of agents {}'.format(NUM_AGENTS))
print('Number of states: {}'.format(len(env.observation_space.sample()[0])))
print('Number of actions: {}'.format(list(env.action_space)[0].n))
class Agent:
    def __init__(self, env, optimizer):
        
        # Initialize atributes
        self._state_size = len(env.observation_space.sample()[0])
        self._action_size = list(env.action_space)[0].n
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1  
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        actions = []
        for agent_idx in range(NUM_AGENTS):
            if np.random.rand() <= self.epsilon:
                return env.action_space.sample()
        
            q_values = self.q_network.predict(state[agent_idx], verbose=0)
            actions.append(np.argmax(q_values[0]))
        return actions

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        for state, action, reward, next_state, terminated in minibatch:
            for agent_idx in range(NUM_AGENTS):
                print(state[agent_idx])
                target = self.q_network.predict(state[agent_idx], verbose=0)
                
                if terminated:
                    target[0][action[agent_idx]] = reward
                else:
                    t = self.target_network.predict(next_state[agent_idx], verbose=0)
                    target[0][action[agent_idx]] = reward + self.gamma * np.amax(t)
                
                self.q_network.fit(state[agent_idx], target, epochs=1, verbose=0)

optimizer = Adam(learning_rate=0.01)
agent = Agent(env, optimizer)

def train_model():
    batch_size = 32
    num_of_episodes = 100
    timesteps_per_episode = 1000
    agent.q_network.summary()
    running_reward_update = .99
    running_reward = 0
    for e in range(0, num_of_episodes):
        # Reset the env
        state = env.reset()
        #state = [np.reshape(state_part, [1,1]) for state_part in state]
        #state = np.reshape(state, [1, 1])

        # Initialize variables
        reward = 0
        terminated = False

        #bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=\
        #    [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        #bar.start()
        for timestep in tqdm (range (timesteps_per_episode), desc="Loading…",  ascii=False, ncols=75):
        #for timestep in range(timesteps_per_episode):
            # Run Action
            action = agent.act(state)
            # Take action
            next_state, reward, terminated, info = env.step(action) 
            reward = reward[0]
            terminated = terminated[0]
            #next_state = [np.reshape(state_part, [1,1]) for state_part in next_state]
            #next_state = np.reshape(next_state, [1, 1])
            running_reward = running_reward_update * running_reward +  (1-running_reward_update) * reward
            agent.store(state, action, reward, next_state, terminated)
            state = next_state
            
            if terminated:
                agent.alighn_target_model()
                break
                
            if len(agent.expirience_replay) > batch_size and timestep % 10 == 0:
                agent.retrain(batch_size)
            
            #if timestep%10 == 0:
            #    bar.update(timestep/10 + 1)
        print(running_reward)
        #bar.finish()
        if (e + 1) % 1 == 0:
            print("**********************************")
            print("Episode: {}".format(e + 1))
            env.render()
            print("**********************************")


train_model()