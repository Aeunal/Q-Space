import gym


################################################################

################################

#env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = gym.make("Breakout-v4")
env.action_space.seed(42)
print(f"Env: {env.unwrapped.spec.id}")
print(f"Obs: {env.observation_space}")



# observation, info = env.reset(seed=42)

# for _ in range(1000):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

#     if terminated or truncated:
#         observation, info = env.reset()


########################################################################
# from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
# observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
# observation_space.sample()

# observation_space = Discrete(4)
# observation_space.sample()

# observation_space = Discrete(5, start=-2)
# observation_space.sample()

# observation_space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
# observation_space.sample()

# observation_space = Tuple((Discrete(2), Discrete(3)))
# observation_space.sample()

# observation_space = MultiBinary(5)
# observation_space.sample()

# observation_space = MultiDiscrete([ 5, 2, 2 ])
# observation_space.sample()


################################################################

from gym.wrappers import RescaleAction

print(f"Action Space: {env.action_space}")
#print(f"Meta Data: {env.metadata}")
# wrapped_env = RescaleAction(env, min_action=0, max_action=1)
# print(f"Scaled action Space: {wrapped_env.action_space}")
# wrapped_env.unwrapped
################################################################

import pygame
from gym.utils.play import play

#mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
#play(env, keys_to_action=mapping)
play(env)

################################

# from gym.utils.play import PlayPlot

# def callback(obs_t, obs_tp1, action, rew, done, info):
#     return [rew,]
# plotter = PlayPlot(callback, 30 * 5, ["reward"])
# env = gym.make("Pong-v0")
# play(env, callback=plotter.callback)


env.close()