# get the neccasary stuff 
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

# list of envs 
num_envs = 3
envs = [lambda: gym.make("BreakoutNoFrameskip-v4") for i in range(num_envs)]

# Vec Env 
envs = SubprocVecEnv(envs)

# render the envs
import time 

# list of envs 
num_envs = 3
envs = [lambda: gym.make("BreakoutNoFrameskip-v4") for i in range(num_envs)]

# Vec Env 
envs = SubprocVecEnv(envs)

init_obs = envs.reset()

for i in range(1000):
    actions = [envs.action_space.sample() for i in range(num_envs)]
    envs.step(actions)
    envs.render()
    time.sleep(0.001)

envs.close()