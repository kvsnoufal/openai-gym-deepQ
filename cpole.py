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

def query_environment(name):
  env = gym.make(name)
  spec = gym.spec(name)
  print(f"Action Space: {env.action_space}")
  print(f"Observation Space: {env.observation_space}")
  print(f"Max Episode Steps: {spec.max_episode_steps}")
  print(f"Nondeterministic: {spec.nondeterministic}")
  print(f"Reward Range: {env.reward_range}")
  print(f"Reward Threshold: {spec.reward_threshold}")

MEMORYLEN=int(10000)
BATCHSIZE=64
EPOCHS=1
# UPDATE_EVERY = 4


class DQNAgent():
    def __init__(self,actions=2,obs=4):
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
import time
from tqdm import tqdm
starttime=time.time()
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
n_episodes=2000
agent=DQNAgent()

max_t=500
eps_start=1.0
eps_end=0.05
eps_decay=0.99



eps = eps_start
env=gym.make('CartPole-v1')
eps_history=[]
for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    for i_ in range(1,max_t+1):
        action = agent.play(state,eps)
        next_state, reward, done, _ = env.step(action)
        if done:
            if (i_>=400):
                agent.step(state, action, reward+5, next_state, done)
            else:
                agent.step(state, action, reward-5, next_state, done)
        else:
            agent.step(state, action, reward, next_state, done)
        
        state = next_state
        score += reward
        if done:
            break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps)
    eps_history.append(eps)
    if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} Epsilon {}'.format(i_episode, np.mean(scores_window),eps))
    if np.mean(scores_window)>=499.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            
            break
endtime= time.time() 
print(endtime-starttime)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("cpole.png")
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(eps_history)), eps_history)
plt.ylabel('Epsilone')
plt.xlabel('Episode #')
plt.savefig("epsilon.png")
# plt.show()  
agent.model.save_weights("cartpole_agent_weights.h5")