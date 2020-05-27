# export DISPLAY=localhost:0.0 
# apt install python-opengl
# apt install ffmpeg
# apt install xvfb
# pip install pyvirtualdisplay
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import gym
from gym.wrappers import Monitor
import glob
import io
import base64

from gym import wrappers
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers,models
import time
from tqdm import tqdm

MEMORYLEN=int(10000)
BATCHSIZE=0
EPOCHS=1

class DQNAgent():
    def __init__(self,actions=4,obs=8):
        self.actions=actions
        self.observations=obs
        self.model=self.load_model()

        self.memory=deque(maxlen=MEMORYLEN)
        self.gamma=0.99
        self.patience=0
        
           
    def play(self,observation,epsilon):
        if (len(self.memory)<BATCHSIZE):
            
            action=np.random.randint(low=0,high=self.actions)
            return action
        else:
            if np.random.random()>epsilon:
#                 print("model")
                action=self.model_predictions(observation)
            else:
                action=np.random.randint(low=0,high=self.actions)
            return action
            
    def step(self,state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        if ((len(self.memory)>=BATCHSIZE) & (np.random.random() < 0.25 )):
            self.train_model()
        pass
 
    
    def train_model(self):
        rnd_indices = np.random.choice(len(self.memory), size=BATCHSIZE)
        data=np.array(self.memory)[rnd_indices]
        # np.random.shuffle(data)
        
        state, action, reward, next_state, done=np.stack(data[:,0]),np.stack(data[:,1]),np.stack(data[:,2]),np.stack(data[:,3]),np.stack(data[:,4])
        qnext_max=np.max(self.model.predict(next_state),axis=1)
        qnext_max=reward+ self.gamma*qnext_max*(1-done)
        qtable_to_update=self.model.predict(state)
        for indx,qs in enumerate(qtable_to_update):
            qtable_to_update[indx,action[indx]]=qnext_max[indx]
        self.model.fit(state,qtable_to_update,epochs=1,verbose=0)
        
        pass
    def model_predictions(self,observation):
        pred=self.model.predict(observation.reshape(1,-1))
        pred=np.argmax(pred)
        return pred
        
    def load_model(self):
        num_input = layers.Input(shape=(self.observations, ))
        x = layers.Dense(24,activation="relu")(num_input)
#         x = layers.BatchNormalization()(x)
#         x = layers.Dropout(0.1)(x)
        x = layers.Dense(24, activation="relu")(x)
#         x = layers.Dropout(0.1)(x)
#         x = layers.BatchNormalization()(x)
        y = layers.Dense(self.actions, activation="linear")(x)
        model = models.Model(inputs=num_input, outputs=y)
        model.compile(loss="mse",optimizer=tf.keras.optimizers.Adam(lr=0.01,decay=0.01))
        model.summary()
        return model

def run(foldername,weight):
    eps=0
    env = gym.make("LunarLander-v2")
    env = gym.wrappers.Monitor(env,foldername)
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    
    scores = []

    agent=DQNAgent()
    agent.model.load_weights(weight)
    
    obs = env.reset()

    frames = []
    cum_reward=0
    while True:
        action = agent.play(obs,eps)

        obs, reward, done, _ = env.step(action)

        cum_reward += reward
        
        if done:
            print(reward,done)
            break
    env.close()    
    print("TOTAL REWARDS: {}".format(cum_reward))

import os
import glob
if __name__ == "__main__":

    
    # run(foldername="vids",weight="./weightsfolder/lunar_agent_weights_trained.h5")
    run(foldername="vids",weight="./weightsfolder/Lunar_weights_100.h5")