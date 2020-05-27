import gym
from gym.wrappers import Monitor
import glob
import io
import base64

from gym import wrappers
from IPython import display
import matplotlib
import matplotlib.pyplot as plt

name="LunarLander-v2"
# name="CarRacing-v0"
env = gym.make(name)


# env = wrappers.Monitor(env, "./gym-results", force=True)
# plt.figure(figsize=(9,9))
# img = plt.imshow(env.render(mode='rgb_array'))
env.reset()
images=[]
rewards=[]
# for _ in range(1000):
while True:    
#     rend=
#     images.append(env.render(mode='rgb_array'))
#     if rend!=None:
#         img.set_data(rend) # just update the data
#         display.display(plt.gcf())
#         display.clear_output(wait=True)
    # env.render()
    # print(observation)
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    
    images.append(observation)
    rewards.append(reward)
    if done: break
env.close()
print(action)
print(observation)