'''
INITIAL CODE FROM https://keras.io/examples/rl/actor_critic_cartpole/, seen here modified for particular env and multi-agent RL
'''
import matplotlib.pyplot as plt
import time
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.api._v2.keras as keras
from keras import layers
#import lbforaging

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000

#env = gym.make('ma_gym:Switch2-v0')
#env = gym.make('ma_gym:Lumberjacks-v0')
env = gym.make('ma_gym:Checkers-v3')
#env = gym.make("Foraging-8x8-{}p-1f-v2".format(NUM_AGENTS))
# Create the environment
#env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
NUM_AGENTS = 2
num_inputs = len(env.observation_space.sample()[0])
num_actions = list(env.action_space)[0].n
num_h1 = 128
num_h2 = 128
num_h3 = 128

print('State space dimension: {}'.format(num_inputs))
print('Action space dimension: {}'.format(num_actions))
print('Number of Agents: {}'.format(NUM_AGENTS))

inputs = layers.Input(shape=(num_inputs,))
common1 = layers.Dense(num_h1, activation="relu")(inputs)
common2 = layers.Dense(num_h2, activation="relu")(common1)
common3 = layers.Dense(num_h2, activation="relu")(common2)
action = layers.Dense(num_actions, activation="softmax")(common3)
critic = layers.Dense(1)(common3)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = [[] for _ in range(NUM_AGENTS)]
critic_value_history = [[] for _ in range(NUM_AGENTS)]
rewards_history = [[] for _ in range(NUM_AGENTS)]
running_reward = 0
episode_count = 0

def run_model():
    state = env.reset()
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0
    while not all(done_n):
        actions = []
        for i in range(NUM_AGENTS):
            agent_state = state[i].copy()
            agent_state = tf.convert_to_tensor(agent_state)
            agent_state = tf.expand_dims(agent_state, 0)
            action_probs, _ = model(agent_state)
            actions.append(np.random.choice(num_actions, p=np.squeeze(action_probs)))
        env.render()
        time.sleep(.1)
        state, reward,  done_n, info = env.step(actions)
        ep_reward += sum(reward)
    print(ep_reward)
    env.close()

inc =0
value_history =[]
while inc < 100000:  # Run until solved
    '''
    if inc % 1000 == 0:
        run_model()
    '''
    inc += 1
    state = env.reset()
    #or env.reset()[0]
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            actions = []
            for i in range(NUM_AGENTS):
                agent_state = state[i].copy()
                agent_state = tf.convert_to_tensor(agent_state)
                agent_state = tf.expand_dims(agent_state, 0)
                # Predict action probabilities and estimated future rewards
                # from environment state
                
                action_probs, critic_value = model(agent_state)
                critic_value_history[i].append(critic_value[0, 0])

                # Sample action from action probability distribution
                
                actions.append(np.random.choice(num_actions, p=np.squeeze(action_probs)))
                action_probs_history[i].append(tf.math.log(action_probs[0, actions[-1]]))
                

            # Apply the sampled action in our environment
            state, reward,  terminated, info = env.step(actions)
            #reward returned as agent-wise list, in this implementation we just give each agent sum of all rewards
            reward = sum(reward)
            for i in range(NUM_AGENTS):
                rewards_history[i].append(reward)
            episode_reward += reward

            if terminated:
                break
        
        value_history.append(episode_reward)
        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        loss_values = []
        for i in range(NUM_AGENTS):
            returns = []
            discounted_sum = 0
            for r in rewards_history[i][::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history[i], critic_value_history[i], returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            loss_values.append(sum(actor_losses) + sum(critic_losses))
        loss_value = sum(loss_values)
        # Backpropagation
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Clear the loss and reward history
        for i in range(NUM_AGENTS):
            action_probs_history[i].clear()
            critic_value_history[i].clear()
            rewards_history[i].clear()
    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break   
model.save('path/to/location')

print(value_history[-10:])
x = np.linspace(0,len(value_history)-1,len(value_history))
plt.plot(value_history)
plt.savefig('value_history plot.pdf')
plt.close()
run_model()


