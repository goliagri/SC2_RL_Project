import numpy as np
import random
import keras.api._v2.keras as keras
from IPython.display import clear_output
from collections import deque
import progressbar
import gym
from keras import layers
from keras import Model, Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam
from tqdm import tqdm

enviroment = gym.make("Taxi-v3").env
enviroment.render()

print('Number of states: {}'.format(enviroment.observation_space.n))
print(enviroment.observation_space)
print('Number of actions: {}'.format(enviroment.action_space.n))

class Agent:
    def __init__(self, enviroment, optimizer):
        
        # Initialize atributes
        self._state_size = enviroment.observation_space.n
        self._action_size = enviroment.action_space.n
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
        if np.random.rand() <= self.epsilon:
            return enviroment.action_space.sample()
        
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            print(state)
            target = self.q_network.predict(state, verbose=0)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t)
                
            self.q_network.fit(state, target, epochs=1, verbose=0)

optimizer = Adam(learning_rate=0.01)
agent = Agent(enviroment, optimizer)

def train_model():
    batch_size = 32
    num_of_episodes = 100
    timesteps_per_episode = 1000
    agent.q_network.summary()
    running_reward_update = .99
    running_reward = 0
    for e in range(0, num_of_episodes):
        # Reset the enviroment
        state = enviroment.reset()
        state = np.reshape(state, [1, 1])

        # Initialize variables
        reward = 0
        terminated = False

        #bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=\
        #    [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        #bar.start()
        
        for timestep in tqdm (range (timesteps_per_episode), desc="Loading…",  ascii=False, ncols=75):
        #for timestep in range(timesteps_per_episode):
            # Run Action
            print(state)
            action = agent.act(state)
            # Take action    
            next_state, reward, terminated, info = enviroment.step(action) 
            next_state = np.reshape(next_state, [1, 1])
            running_reward = running_reward_update * running_reward +  (1-running_reward_update) * reward
            agent.store(state, action, reward, next_state, terminated)
            state = next_state
            
            if terminated:
                agent.alighn_target_model()
                break
                
            if len(agent.expirience_replay) > batch_size and timestep % batch_size == 0:
                agent.retrain(batch_size)
            
            #if timestep%10 == 0:
            #    bar.update(timestep/10 + 1)
        print(running_reward)
        #bar.finish()
        if (e + 1) % 10 == 0:
            print("**********************************")
            print("Episode: {}".format(e + 1))
            enviroment.render()
            print("**********************************")


train_model()