import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import EvalCallback
import os
import random


from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()


from matplotlib import pyplot as plt, animation
%matplotlib inline
from IPython import display

def create_anim(frames, dpi, fps):
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
    patch = plt.imshow(frames[0])
    def setup():
        plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, init_func=setup, frames=len(frames), interval=fps)
    return anim

def display_anim(frames, dpi=72, fps=60):
    anim = create_anim(frames, dpi, fps)
    return anim.to_jshtml()

def save_anim(frames, filename, dpi=72, fps=60):
    anim = create_anim(frames, dpi, fps)
    anim.save(filename)


class trigger:
    def __init__(self):
        self._trigger = True

    def __call__(self, e):
        return self._trigger

    def set(self, t):
        self._trigger = t


def get_env_id(env):
    if isinstance(env, str):
        return env
    if hasattr(env, 'envs'):
        return env.envs[0].unwrapped.spec.id
    else:
        return env.unwrapped.spec.id
    
def test_env(env, model=None, episodes=5):
    if model == None:
        env_id = get_env_id(env)
        env = gym.make(env_id)
    frames = []
    for episode in range(1, episodes+1):
        obs = env.reset()  #state = env.reset()
        done = False
        score = 0
        while not done:
            frames.append(env.render(mode = 'rgb_array'))
            if model == None:
                action = env.action_space.sample()
            else:
                action , _ = model.predict(obs) #action = env.action_space.sample()
            obs, reward, done, info = env.step(action) #n_state, reward, done, info = env.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode,score))
    env.close()
    return frames

def save_frames(new_frames, name='new'):
    filename = f'Training/Videos/Breakout-v0_{name}'
    save_anim(new_frames, filename=filename+".avi")
    save_anim(new_frames, filename=filename+".gif")
    return filename

def show_vid(filename, ext=".gif"):
    return display.Video(filename+ext, width=300, height=400)

def anim_frames(frames):
    return display.HTML(display_anim(frames))


def make_env_vec(env_id, n=4, stack=4):
    env = make_atari_env(env_id, n_envs=n, seed=0)  #make_atari_env helps in creating wrapped Atari env
    env = VecFrameStack(env, n_stack=stack)             #VecFrameStack allows to stack env together
    return env

def make_model(env, policy='CnnPolicy'):
    LOG_PATH = os.path.join('Training','Logs')
    model    = PPO(policy, env, verbose=1, tensorboard_log=LOG_PATH)
    return model

def save_model(model, overwrite=False):
    if overwrite:
        MODEL_PATH = os.path.join('Training','Saved Models','PPO_Best_Atari_Model')
    else:
        r_id = random.randint(0,999999)
        r_id = str(r_id).zfill(6)
        MODEL_PATH = os.path.join('Training','Saved Models',f'PPO_Atari_Model_{r_id}')
    #atari_path = os.path.join('Training','Saved Models','PPO_Atari_Model')
    model.save(MODEL_PATH)
    return MODEL_PATH

def evaluate(env, model, n=50, render=False, log=False):
    ev = evaluate_policy(model, env, n_eval_episodes=n, render=render)
    env.close()
    if log: print(ev)
    return ev

ENVIRONMENT_NAME = 'Breakout-v0'    
env = gym.make(ENVIRONMENT_NAME)

anim_frames(test_env(env))

env = make_env_vec(ENVIRONMENT_NAME)
model_mlp = make_model(env, "MlpPolicy")
model_mlp_5e4 = model_mlp.learn(total_timesteps=50000)
path_mlp_5e4 = save_model(model_mlp_5e4)


env = make_env_vec(ENVIRONMENT_NAME, n=1)
evaluate(env, model_mlp_5e4)

frames = test_env(env, model_mlp_5e4)
file_mlp = save_frames(frames, "MLP")
anim_frames(frames)
