from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI
import numpy as np
import get_features
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.api._v2.keras as keras
from keras import layers

#########################################################################################################################
'''                              AI Code                                     '''
#########################################################################################################################
#constants
NUM_ENEMIES_CONSIDERED = 1
NUM_AGENTS = 12

class ClosestNLingsBot(BotAI):

    async def on_start(self):
        self.actions_probs_history = [[] for _ in range(NUM_AGENTS)]
        self.rewards_history = [[] for _ in range(NUM_AGENTS)]
        self.num_agents_history = []
        self.critic_value_history = [[] for _ in range(NUM_AGENTS)]
        self.NUM_ACTIONS = 2 + NUM_ENEMIES_CONSIDERED
        self.prev_value = 650
        self.client.game_step: int = 50
        self.episode_reward = 0
        self.prev_num_agents = 0
        self.tag2AgentIdx = {}
        agent_tags = list(self.workers.tags)
        for i,tag in enumerate(agent_tags):
            self.tag2AgentIdx.update({tag:i})
        self.prev_tags = self.workers.tags
        
    async def on_end(self, result):
        '''
        self.rewards_history = self.rewards_history[1:]
        self.actions_probs_history = self.actions_probs_history[:-self.num_agents_history[-1]]
        self.critic_value_history = self.critic_value_history[:-self.num_agents_history[-1]]
        self.num_agents_history = self.num_agents_history[:-1]
        '''
        just_died = self.prev_tags.difference(self.workers.tags)
        for tag in just_died:
            self.rewards_history[self.tag2AgentIdx[tag]].append(self.get_state_value()-self.prev_value)
        
        print(self.rewards_history)
        print(self.critic_value_history)
        print(self.actions_probs_history)

    async def on_step(self, iteration: int):
        cur_value = self.get_state_value()
        prev_reward = cur_value - self.prev_value
        self.prev_value = cur_value
        for worker in self.workers:
            agentIdx = self.tag2AgentIdx[worker.tag]
            #get list of closest n lings
            enemies_considered = self.enemy_units.closest_n_units(worker, NUM_ENEMIES_CONSIDERED)
            enemies_considered.sort(key=worker.distance_to)
            #Get state (inputs) and plug it into model
            state = self.get_state_feats(worker, enemies_considered)
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            action_probs, critic_value = model(state)
            #action_probs = np.array([0.34,0.33,0.33]).reshape(1,-1)
            #critic_value = np.ones((1,1))*iteration
            #Sample returned probabilities to choose action
            action = np.random.choice(self.NUM_ACTIONS, p=np.squeeze(action_probs))
            if action == 0:
                #If already mining, do nothing, elif carrying resource, return it, else gather from closest mineral patch
                self.mine(worker)
            elif action == 1:
                #Move in opposite direction from closest zergling
                self.run(worker, enemies_considered)
            elif action >= 2 and action < 2+NUM_ENEMIES_CONSIDERED:
                #Attack given ling
                self.attack(worker, enemies_considered, action)
            else:
                raise Exception('Invalid action given ({})'.format(action))

            #Update histories agent-wise
            if iteration != 0:
                self.rewards_history[agentIdx].append(prev_reward)
                self.critic_value_history[agentIdx].append(critic_value[0, 0])
            self.actions_probs_history[agentIdx].append(tf.math.log(action_probs[0, action]))
        
        self.episode_reward += prev_reward
        self.num_agents_history.append(len(list(self.workers)))

        #We need to add last reward to agents that died between last and current state
        just_died = self.prev_tags.difference(self.workers.tags)
        for tag in just_died:
            self.rewards_history[self.tag2AgentIdx[tag]].append(prev_reward)
        self.prev_tags = self.workers.tags

    def mine(self, worker):
        if not worker.is_collecting:
            if worker.is_carrying_minerals:
                worker.return_resource()
            else:
                field = self.mineral_field.closest_to(worker)
                worker.gather(field)

    def attack(self, worker, enemies_considered, action):
        if len(list(enemies_considered)) > action-2:
            worker.attack(list(enemies_considered)[action-2])

    def run(self, worker, enemies_considered):
        None

    def get_state_value(self):
        return self.minerals + 50 * len(list(self.units))

    def get_state_feats(self, worker, enemies_considered):
        '''
        All enemy/allies distances, health, shield, self health, shield, has minerals
        '''
        feats = []

        feats.extend([worker.health, worker.shield, int(worker.is_carrying_resource), worker.position[0], worker.position[1]])
        for enemy in enemies_considered:
            feats.extend([enemy.health,enemy.distance_to(worker)])
        for _ in range(0, NUM_ENEMIES_CONSIDERED - len(enemies_considered)):
            feats.extend([0,100])
        
        return feats

#########################################################################################################################
'''                              Model Code                                     '''
#########################################################################################################################
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 5 + NUM_ENEMIES_CONSIDERED*2
num_actions = 2 + NUM_ENEMIES_CONSIDERED
num_hidden = 32

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])


#########################################################################################################################
'''                              RL Code                                    '''
#########################################################################################################################

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
#critic_value_history = []
running_reward = 0
episode_count = 0

def run_actor_critic():
    while True:
        bot = ClosestNLingsBot()
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            run_game(maps.get("MineAndKillZerglings"), [Bot(Race.Protoss, bot)], realtime=False)
        
        loss_values = []
        for i in range(NUM_AGENTS):
            returns = []
            rewards_history = bot.rewards_history[i]
            actions_probs_history = bot.actions_probs_history[i]
            critic_value_history = bot.critic_value_history[i]
            discounted_sum = 0
            #for i, r in enumerate(bot.rewards_history[::-1]):
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                #for _ in range(bot.num_agents_history[-i]):
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            #running_reward = 0.05 * bot.episode_reward + (1 - 0.05) * running_reward

            # Calculating loss values to update our network
            history = zip(actions_probs_history, critic_value_history, returns)
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

        # Backpropagation
        #actor_loss = sum(actor_losses)
        #critic_loss = sum(critic_losses)
        loss_value = sum(loss_values)
        print(loss_value)
        grads = tape.gradient(loss_value, model.trainable_variables)
        #grads = tape.gradient(returns[-1], model.trainable_variables)
        print(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        #action_probs_history.clear()
        #critic_value_history.clear()
        #rewards_history.clear()

run_actor_critic()