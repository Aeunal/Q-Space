import gym
import random
import time
import numpy as np

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        # Normalise observation by 255
        return obs / 255.0

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # Clip reward between 0 to 1
        return np.clip(reward, 0, 1)
    
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        if action == 3:
            return random.choice([-1, -1, 1])
        else:
            return action

#env = gym.make("ALE/Breakout-v5", render_mode="human")
env = gym.make("Breakout-v4")
wrapped_env = ObservationWrapper(RewardWrapper(ActionWrapper(env)))

epoch = 10
while epoch >= 0:
    epoch -= 1

    obs = wrapped_env.reset()

    for step in range(500):
        action = wrapped_env.action_space.sample()
        obs, reward, done, info = wrapped_env.step(action)
        if done: break
        
        # Raise a flag if values have not been vectorised properly
        if (obs > 1.0).any() or (obs < 0.0).any():
            print("Max and min value of observations out of range")
        
        # Raise a flag if reward has not been clipped.
        if reward < 0.0 or reward > 1.0:
            assert False, "Reward out of bounds"
        
        # Check the rendering if the slider moves to the left.
        wrapped_env.render(mode="rgb_array")
        
        time.sleep(0.001)

wrapped_env.close()

print("All checks passed")