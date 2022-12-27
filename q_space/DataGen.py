import tensorflow as tf
import numpy as np


def autoencoder_data_generators(im_size, batch_size=32, val_split = 0.5):
    train_ds = tf.keras.utils.image_dataset_from_directory(
    "./images",
    validation_split=val_split,
    subset="training",
    seed=123,
    image_size=im_size,
    color_mode='grayscale',
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    "./images",
    validation_split=val_split,
    subset="validation",
    seed=123,
    image_size=im_size,
    color_mode='grayscale',
    batch_size=batch_size)

    return train_ds, val_ds

def normalize_datagen(dg, crop=None):
    in_shape = dg._structure[0].shape
    #in_size = np.prod(in_shape[1:])
    in_size = np.prod(crop[2:])
    batch_size = dg._batch_size#.numpy()

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    def change_inputs(images, labels):
        x = normalization_layer(images)
        #x = tf.image.resize(x, [28, 28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        if crop:
            x = tf.image.crop_to_bounding_box(
                x, *crop
            )

        # flatten
        print(x.shape)
        batch = x.shape[0] if x.shape[0] else batch_size
        x = tf.reshape(x, [batch,in_size])
        return x, x

    normalized_dg = dg.map(change_inputs)
    return normalized_dg

# 16(top)+23(mid)+2(bot) = 43 (all) - height ; 36 - width
def crop_height(input_size, include_top=True, include_mid=True, include_bot=True, top_len=16, bot_len=2):
    crop_top = input_size[0]
    crop_bot = input_size[0]
    if include_top:
        crop_top = 0
    if include_mid:
        crop_top = min(crop_top, top_len)
    if include_bot:
        crop_top = min(crop_top, input_size[0]-bot_len)
    
    if include_bot:
        crop_bot = 0
    if include_mid:
        crop_bot = min(crop_bot, bot_len)
    if include_top:
        crop_bot = min(crop_bot, input_size[0]-top_len)
    
    return (crop_top, 0, input_size[0]-crop_bot-crop_top, input_size[1])

    #
    # top
    # autoencoder, encoder, decoder = Model.make_auto_encoder((16,36))

    # mid + bot
    # autoencoder, encoder, decoder = Model.make_auto_encoder((25,36))

    # bot
    # autoencoder, encoder, decoder = Model.make_auto_encoder((2,36))

    # only mid
    # autoencoder, encoder, decoder = Model.make_auto_encoder((23,36))

