from q_space import Policy 
from q_space import Environment
from q_space import DataGen
from q_space import Model
from q_space import Logging
import time
import numpy as np
import random

def Generate_Autoencoder_Datagen(crop):
    train_ds, val_ds = DataGen.autoencoder_data_generators(Environment.OBS_SHAPE)
    print("Train batches:", (train_ds.cardinality()))
    train_ds = DataGen.normalize_datagen(train_ds, crop=crop)
    val_ds = DataGen.normalize_datagen(val_ds, crop=crop)
    return train_ds, val_ds

def Build_AutoEncoder_Experiment(name="full_height", include_areas = [True, True, True], epoch=500, encoding_size=32):
    crop = DataGen.crop_height(Environment.OBS_SHAPE, *include_areas)
    encoder_in = crop[2:]
    print("Encoder Input: ", encoder_in)
    print("Cropping images with: ", crop)

    # full
    # autoencoder, encoder, decoder = Model.make_auto_encoder(Environment.OBS_SHAPE)
    autoencoder, encoder, decoder = Model.make_auto_encoder(encoder_in, encoding_dim=encoding_size)

    train_ds, val_ds = Generate_Autoencoder_Datagen(crop)

    Model.fit(autoencoder, train_ds, val_ds, epoch=epoch, batch=256)
    Model.Save(autoencoder, name)

    return (autoencoder, encoder, decoder), (train_ds, val_ds), crop


def Load_AutoEncoder(name):
    # TODO: find the most recent name
    autoencoder, img_shape = Model.Load(name)
    encoder, decoder = Model.extract_encoder_decoder(autoencoder)
    return (autoencoder, encoder, decoder), img_shape

def Generate_Random_AutoEncoder_Data(n):
    random_policy = Policy.RandomPolicy()
    TrainPolicy(random_policy, n)

def TrainPolicy(policy, epochs):
    env = Environment.make_env()
    obs = Environment.reset(env)
    print(obs.shape)

    for i in range(epochs):
        act = policy(obs)
        obs, reward, done = Environment.step(env, act)

        if done:
            obs = Environment.reset(env)

        policy.update(obs, act, reward, done)
        #img1 = PIL.Image.fromarray(obs_g)
        #clear_output(wait=False)
        #display(img1.resize((200,200)))
        #img1.save(f"images/test{i}.png")
        #time.sleep(0.1)
        #clear_output(wait=False)
    Environment.close(env)


def RenderPolicyNormal(policy, iterations=100):

    Logging.enable_display(str(random.randint(0, 100)))
    env = Environment.make_env()
    obs = Environment.reset(env)
    print(obs.shape)

    for i in range(iterations):
        act = policy(obs)
        obs, reward, done = Environment.step(env, act, save=False)

        if done:
            obs = Environment.reset(env)

        #print(obs)
        #print(decoded_obs_all.shape)
        Logging.try_displaying(obs)
        
        if reward != 0:
            print("\rReward: ", reward, end="")
        
        policy.update(obs, act, reward, done)
            
        time.sleep(0.1)

    Environment.close(env)

from q_space.Environment import Obs_Cropper

def encode(obs, encoder):
    flatten_obs = obs.reshape(1, np.prod(obs.shape))
    encoded_obs = encoder(flatten_obs)
    return encoded_obs

def multi_encode(obs, encoders, shapes):
    encoder_top, encoder_mid, encoder_bot = encoders
    top_shape, mid_shape, bot_shape = shapes
    obs_top = Obs_Cropper(obs, top_shape)
    obs_mid = Obs_Cropper(obs, mid_shape)
    obs_bot = Obs_Cropper(obs, bot_shape)
    flatten_top = obs_top.reshape(1, np.prod(obs_top.shape))
    flatten_mid = obs_mid.reshape(1, np.prod(obs_mid.shape))
    flatten_bot = obs_bot.reshape(1, np.prod(obs_bot.shape))
    encoded_top = encoder_top(flatten_top)
    encoded_mid = encoder_mid(flatten_mid)
    encoded_bot = encoder_bot(flatten_bot)
    encoded_obs = np.hstack((encoded_top, encoded_mid, encoded_bot))
    return encoded_obs

def RenderPolicyEncoder(policy, encoder, decoder, iterations=100, encoding_threshold = 0.2):

    Logging.enable_display(str(random.randint(0, 100)))
    env = Environment.make_env()
    obs = Environment.reset(env)
    print(obs.shape)

    for i in range(iterations):
        act = policy(obs)
        obs, reward, done = Environment.step(env, act, save=False)

        if done:
            obs = Environment.reset(env)

        #print(obs)
        #obs_all = Obs_Cropper(obs, all_shape)
        encoded_obs = encode(obs, encoder)
        decoded_obs = decoder(encoded_obs)
        decoded_obs = decoded_obs > encoding_threshold
        decoded_obs = decoded_obs.numpy().reshape((obs.shape))
        #print(decoded_obs.shape)
        #Logging.try_displaying(obs)
        Logging.try_displaying(decoded_obs)
        
        if reward != 0:
            print("\rReward: ", reward, end="")
        
        policy.update(obs, act, reward, done)
            
        time.sleep(0.1)

    Environment.close(env)


def RenderPolicyEncoders(policy, encoders, decoders, input_shapes, iterations=100, encoding_threshold = 0.2):

    encoder_top, encoder_mid, encoder_bot = encoders
    decoder_top, decoder_mid, decoder_bot = decoders
    top_shape, mid_shape, bot_shape = input_shapes

    Logging.enable_display(str(random.randint(0, 100)))
    env = Environment.make_env()
    obs = Environment.reset(env)
    print(obs.shape)

    for i in range(iterations):
        act = policy(obs)
        obs, reward, done = Environment.step(env, act, save=False)

        if done:
            obs = Environment.reset(env)

        #print(obs)

        # obs_all = crop_obs(obs, all_shape)
        obs_top = Obs_Cropper(obs, top_shape)
        obs_mid = Obs_Cropper(obs, mid_shape)
        obs_bot = Obs_Cropper(obs, bot_shape)

        # flatten_all = obs_all.reshape(1, np.prod(obs_all.shape))
        flatten_top = obs_top.reshape(1, np.prod(obs_top.shape))
        flatten_mid = obs_mid.reshape(1, np.prod(obs_mid.shape))
        flatten_bot = obs_bot.reshape(1, np.prod(obs_bot.shape))
        
        # encoded_all = encoder_all(flatten_all)
        encoded_top = encoder_top(flatten_top)
        encoded_mid = encoder_mid(flatten_mid)
        encoded_bot = encoder_bot(flatten_bot)

        encoded_merged = np.hstack((encoded_top, encoded_mid, encoded_bot))
        #print("Merged Encoding:", encoded_merged.shape)

        #encoded_top, encoded_mid, encoded_bot = np.hsplit(encoded_merged, 3)
        split_indices = [encoded_merged.shape[1]//2, encoded_merged.shape[1]-(encoded_merged.shape[1]//4)]
        encoded_top, encoded_mid, encoded_bot = np.hsplit(encoded_merged, split_indices) # Weighted split

        # decoded_all = decoder_all(encoded_all)
        decoded_top = decoder_top(encoded_top)
        decoded_mid = decoder_mid(encoded_mid)
        decoded_bot = decoder_bot(encoded_bot)
        
        # decoded_all = decoded_all > encoding_threshold
        decoded_top = decoded_top > encoding_threshold
        decoded_mid = decoded_mid > encoding_threshold
        decoded_bot = decoded_bot > encoding_threshold
        
        # decoded_obs_all = decoded_obs_all.numpy().reshape((obs_all.shape))
        decoded_obs_top = decoded_top.numpy().reshape((obs_top.shape))
        decoded_obs_mid = decoded_mid.numpy().reshape((obs_mid.shape))
        decoded_obs_bot = decoded_bot.numpy().reshape((obs_bot.shape))
        #print(decoded_obs_all.shape)

        decoded_obs_merged = np.vstack((decoded_obs_top, decoded_obs_mid, decoded_obs_bot))

        #Logging.try_displaying(obs)
        Logging.try_displaying(decoded_obs_merged)
        
        if reward != 0:
            print("\rReward: ", reward, end="")
        
        policy.update(obs, act, reward, done)
            
        time.sleep(0.1)

    Environment.close(env)


def TrainPolicyEncoders(policy, encoders, decoders, input_shapes, iterations=100, encoding_threshold = 0.2, save=False, render=False):

    #encoder_top, encoder_mid, encoder_bot = encoders
    #top_shape, mid_shape, bot_shape = input_shapes
    
    if render:
        Logging.enable_display(str(random.randint(0, 100)))
    env = Environment.make_env()
    obs = Environment.reset(env)
    print(obs.shape)
    encoded_obs = multi_encode(obs, encoders,input_shapes)
    print(encoded_obs.shape)
    act = policy(encoded_obs)

    for i in range(iterations):

        obs, reward, done = Environment.step(env, act, save=save)
        prev_encoded_obs = encoded_obs
        encoded_obs = multi_encode(obs, encoders,input_shapes)
        #print("Merged Encoding:", encoded_obs.shape)
        prev_act = act
        act = policy(encoded_obs)

        policy.update(prev_encoded_obs, prev_act, reward, encoded_obs, act)

        if done:
            obs = Environment.reset(env)

        if render:
            Logging.try_displaying(obs)
        
        if reward != 0:
            print("\rReward: ", reward, end="")
        
            
        time.sleep(0.1)

    Environment.close(env)


    
def TrainPolicyEnv(env, policy, iterations=100, encoding_threshold = 0.2, save=False, render=False):

    #encoder_top, encoder_mid, encoder_bot = encoders
    #top_shape, mid_shape, bot_shape = input_shapes
    
    if render:
        Logging.enable_display(str(random.randint(0, 100)))
    obs = env.reset()
    print(obs.shape)
    act = policy(obs)

    for i in range(iterations):

        obs, reward, done, info = env.step(act, save=save)#, show=render)
        prev_obs = obs
        #print("Merged Encoding:", encoded_obs.shape)
        prev_act = act
        act = policy(obs)

        policy.update(prev_obs, prev_act, reward, obs, act)

        if done:
            obs = env.reset()

        if render:
            Logging.try_displaying(obs)
        
        if reward != 0:
            print("\rReward: ", reward, end="")
        
            
        time.sleep(0.1)

    env.close()