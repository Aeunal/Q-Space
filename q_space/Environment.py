import gym
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import PIL
import random

def randomize_id():
    return random.randint(100_000,999_999)

Obs_Cropper = lambda orig_obs, crop_shape: orig_obs[crop_shape[0]:crop_shape[0]+crop_shape[2],crop_shape[1]:crop_shape[1]+crop_shape[3]]

def multi_encode(obs, encoders, shapes):
    encoder_top, encoder_mid, encoder_bot = encoders
    top_shape, mid_shape, bot_shape = shapes
    obs_top = Obs_Cropper(obs, top_shape)
    obs_mid = Obs_Cropper(obs, mid_shape)
    obs_bot = Obs_Cropper(obs, bot_shape)
    flatten_top = obs_top.reshape(1, np.prod(obs_top.shape))
    flatten_mid = obs_mid.reshape(1, np.prod(obs_mid.shape))
    flatten_bot = obs_bot.reshape(1, np.prod(obs_bot.shape))

    #encoded_top = encoder_top(flatten_top)
    encoded_top = encoder_top.evaluate(flatten_top)
    #encoded_mid = encoder_mid(flatten_mid)
    encoded_mid = encoder_mid(flatten_mid)
    #encoded_bot = encoder_bot(flatten_bot)
    encoded_bot = encoder_bot(flatten_bot)
    
    encoded_obs = np.hstack((encoded_top, encoded_mid, encoded_bot))
    return encoded_obs

class EncoderWrapper(gym.Wrapper):
    def __init__(self, env, encoders, encode_shapes):
        super().__init__(env)
        self.env = env
        self.encoders = encoders
        self.encode_shapes = encode_shapes
        self.encode_dims = [encoder.output_shape[1] for encoder in encoders]

        self.observation_space = gym.spaces.Box(low=np.float32(0.0), high=np.inf, 
            shape=(sum(self.encode_dims),), dtype=np.float32) 
        self.action_space = gym.spaces.Discrete(3)
        
        self.LIVE = 5
        self.RANDOM_PREFIX = randomize_id()
        self.PNG_IDX = 0
        
    def get_action_meanings(self):
        return ['NOOP & FIRE', 'LEFT', 'RIGHT']

    def step(self, action, show=False, save=False):
        next_state, reward, done, info = self.env.step(action)
        reward += 0.01
        current_live = info['ale.lives']
        if current_live < self.LIVE:
            reward -= 5
            self.LIVE = current_live
        
        next_state = pre_process_obs(next_state)

        next_state = multi_encode(next_state, self.encoders, self.encode_shapes)

        if save:
            save_obs(next_state, show=show, postvals=[
                RANDOM_PREFIX, PNG_IDX, reward, current_live, (1 if done else 0)
            ])
            PNG_IDX += 1
            if done:
                RANDOM_PREFIX = randomize_id()
                PNG_IDX = 0

        return next_state, reward, done, info

    def reset(self, show=False):
        self.LIVE = 5
        obs = self.env.reset()
        obs = pre_process_obs(obs)
        obs = multi_encode(obs, self.encoders, self.encode_shapes)
        return obs

    def render(self):
        return self.env.render()

    def close(self):
        self.LIVE = 5
        self.env.close()


class ProcessedWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.observation_space = gym.spaces.Box(low=np.bool_(False), high=np.bool_(True), 
            shape=(41,36,1), dtype=np.bool_) 
        self.action_space = gym.spaces.Discrete(3)
        
        self.LIVE = 5
        self.RANDOM_PREFIX = randomize_id()
        self.PNG_IDX = 0
        
    def get_action_meanings(self):
        return ['NOOP & FIRE', 'LEFT', 'RIGHT']

    def step(self, action, show=False, save=False):
        next_state, reward, done, info = self.env.step(action)
        reward += 0.01
        current_live = info['ale.lives']
        if current_live < self.LIVE:
            reward -= 5
            self.LIVE = current_live
        
        next_state = pre_process_obs(next_state)

        if save:
            save_obs(next_state, show=show, postvals=[
                RANDOM_PREFIX, PNG_IDX, reward, current_live, (1 if done else 0)
            ])
            PNG_IDX += 1
            if done:
                RANDOM_PREFIX = randomize_id()
                PNG_IDX = 0

        return next_state, reward, done, info

    def reset(self, show=False):
        self.LIVE = 5
        obs = self.env.reset()
        obs = pre_process_obs(obs)
        return obs

    def render(self):
        return self.env.render()

    def close(self):
        self.LIVE = 5
        self.env.close()


RANDOM_PREFIX = randomize_id()
PNG_IDX = 0
SEED = 42

def explain_configure_env(env):
    # mode=None, difficulty=None, obs_type='ram', frameskip=(2, 5), repeat_action_probability=0.0, full_action_space=False
    env.seed(SEED)
    env.action_space.seed(SEED)
    print(f"Env: {env.unwrapped.spec.id}")
    print(f"Obs: {env.observation_space}")
    print(f"Actions: {env.action_space} - {env.get_action_meanings()}")
    print(f"Reward range: {env.reward_range}")
    #print(f"Meta Data: {env.metadata}")

def make_encoded_env(encoders, encode_shapes, environment_name='BreakoutNoFrameskip-v4'):
    env = EncoderWrapper(gym.make(environment_name), encoders, encode_shapes) 
    explain_configure_env(env)
    return env
    
def make_processed_env(environment_name='BreakoutNoFrameskip-v4'):
    env = ProcessedWrapper(gym.make(environment_name)) 
    explain_configure_env(env)
    return env

def make_env(environment_name='BreakoutNoFrameskip-v4'):
    #env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = gym.make(environment_name) 
    explain_configure_env(env)
    return env

    '''
    env.env.ale.saveScreenPNG('images/test_image.png')
    plt.imshow(resize(obs))
    skimage.measure.block_reduce(obs_g, (3,3), np.max).shape
    obs_g = np.average(obs,2)
    plt.imshow(obs_g)
    plt.imshow(np.average(obs,2))
    def resize(obs):
        new_obs = cv2.resize(obs, dsize=(320, 210), interpolation=cv2.INTER_NEAREST)
        new_obs = cv2.resize(new_obs, dsize=(80, 50), interpolation=cv2.INTER_NEAREST)
        return new_obs
    cv2.imwrite("images/test1.png", img1)
    '''

from IPython.display import display, clear_output, update_display
OBS_SHAPE = (41,36) # preprocessed size
SCREEN_FILLED = False

def save_obs(obs, postvals=[], show=False):
    global SCREEN_FILLED
    img = PIL.Image.fromarray(obs)
    postfix = "_".join(map(str,map(int,postvals)))
    img.save(f"images/training/test_{postfix}.png")

    #plt.imshow(np.hstack((obs_g, obs_g & (obs_g ^ prev_obs_g))).astype(np.bool_))
    #plt.imshow(obs_g)
    if show:
        img = img.resize((200,200))
        if SCREEN_FILLED:
            update_display(img, id = "render")
        else:
            SCREEN_FILLED = True
            #clear_output(wait=False)
            display(img, id = "render")
    else:
        if SCREEN_FILLED:
            #clear_output(wait=False)
            SCREEN_FILLED = False

def pre_process_obs(obs):
    # prev_obs_g = np.zeros((41,36), dtype=np.bool_)
    obs_g = np.average(obs,2)
    obs_g = obs_g.astype(np.bool_)
    obs_g = measure.block_reduce(obs_g, (4,4), np.max)
    obs_g = obs_g[8:-4,2:-2]
    #prev_obs_g = obs_g
    #obs_g = np.reshape(obs_g, (*obs_g.shape,1))
    return obs_g


LIVE = 5
def step(env, action, show=False, save=True):
    global LIVE, RANDOM_PREFIX, PNG_IDX
    obs, reward, done, info = env.step(action)
    reward += 0.01
    current_live = info['ale.lives']
    if current_live < LIVE:
        reward -= 5
        LIVE = current_live
    
    obs = pre_process_obs(obs)

    if save:
        save_obs(obs, show=show, postvals=[
            RANDOM_PREFIX, PNG_IDX, reward, current_live, (1 if done else 0)
        ])
        PNG_IDX += 1
        if done:
            RANDOM_PREFIX = randomize_id()
            PNG_IDX = 0

    return obs, reward, done

    #obs, *_ = env.step(1)
    #env.render()

def reset(env, show=False):
    global LIVE
    LIVE = 5
    obs = env.reset()
    obs = pre_process_obs(obs)
    return obs


def render(env):
    env.render()

def close(env):
    global LIVE
    LIVE = 5
    env.close()


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

# from gym.wrappers import RescaleAction

# wrapped_env = RescaleAction(env, min_action=0, max_action=1)
# print(f"Scaled action Space: {wrapped_env.action_space}")
# wrapped_env.unwrapped
################################################################

# import pygame
# from gym.utils.play import play

#mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
#play(env, keys_to_action=mapping)
# play(env)

################################

# from gym.utils.play import PlayPlot

# def callback(obs_t, obs_tp1, action, rew, done, info):
#     return [rew,]
# plotter = PlayPlot(callback, 30 * 5, ["reward"])
# env = gym.make("Pong-v0")
# play(env, callback=plotter.callback)
