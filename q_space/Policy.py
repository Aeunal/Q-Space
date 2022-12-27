import numpy as np
import random


class BasePolicy():
    def __init__(self, state_space=None):
        self.state_space = state_space
        self.history = []
        pass
    
    def act(self, obs):
        pass

    def update(self, obs, act, reward, done):
        pass

    def update(self, obs, act, reward, next_obs, next_act):
        pass

    def __call__(self, obs):
        return self.act(obs)

class RandomPolicy(BasePolicy):

    def act(self, obs):
        return random.randint(1,3)

class DebuggingPolicy(BasePolicy):

    def act(self, obs):
        #print(obs.shape)
        return random.randint(1,3)

    def update(self, obs, act, reward, next_obs, next_act):
        self.history.append([obs, act, reward, next_obs, next_act])


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

class MLP_Policy(BasePolicy):
    def __init__(self,):
        super(MLP_Policy, self).__init__()

        states = env.observation_space.shape
        actions = env.action_space.n
        self.build_model(states, actions)    

    def build_model(self, states, actions):
        model = Sequential()    
        model.add(Dense(24, activation='relu', input_shape=states))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model