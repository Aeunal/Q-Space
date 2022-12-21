import sys
import re
import multiprocessing
import os.path as osp
import tensorflow.compat.v1 as tf

import baselines.a2c as a2c
from baselines import logger
import numpy as np
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.tf_util import get_session
import gym
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env

from importlib import import_module

from collections import defaultdict




num_timesteps = 1e6
alg = 'deepq'#'ppo2'
save_video_interval = False
save_video_length = 1.0
network = None # mlp, cnn, lstm, cnn_lstm, conv_only



_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

def get_env_type(env_id, env_type=None):
    if env_type is not None:
        return env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id

def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'
def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
    return alg_module
def get_learn_function(alg):
    return get_alg_module(alg).learn
def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def build_env(env_id):
    global alg
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = 1#ncpu
    seed = 42

    env_type, env_id = get_env_type(env_id)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=None, reward_scale=1.0)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, nenv or 1, seed, reward_scale=1.0, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def train(env_id, ):
    env_type, env_id = get_env_type(env_id)
    print('env_type: {}'.format(env_type))


    total_timesteps = int(num_timesteps)
    seed = 42

    learn = get_learn_function(alg)
    alg_kwargs = get_learn_function_defaults(alg, env_type)
    #alg_kwargs.update(extra_args)

    env = build_env(env_id)
    if save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % save_video_interval == 0, video_length=save_video_length)

    if network:
        alg_kwargs['network'] = network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


model, env = train("Breakout-v4")

save_path = r"C:/Users/aeunal/Pictures/itü/ai/Project/save/"
model.save(save_path)

play = True

if play:
    logger.log("Running trained model")
    obs = env.reset()
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))
    episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)
        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('episode_rew={}'.format(episode_rew[i]))
                episode_rew[i] = 0
env.close()