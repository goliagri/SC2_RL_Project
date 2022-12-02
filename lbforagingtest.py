import gym
import rware
import lbforaging
#env = gym.make("rware:rware-tiny-2ag-v1")
env = gym.make("rware-tiny-2ag-v1", sensor_range=3, request_queue_size=6)
#NUM_AGENTS = 0
#env = gym.make("Foraging-8x8-{}p-1f-v2".format(NUM_AGENTS))

state = env.reset()[0]
env.render()
print(state)

