import numpy as np
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.callbacks import Callback
from keras import Input, Model
import tensorflow as tf
from tensorflow.keras import regularizers

from tensorflow.keras.models import load_model


def make_auto_encoder(img_input, encoding_dim = 32, hidden_dim = 0, regularization=False):
    in_size = np.prod(img_input)    
    input_img = Input(shape=(in_size,))

    kernel_reg = None
    bias_reg = None
    activity_reg = None
    if regularization:
        kernel_reg = regularizers.L1L2(l1=1e-6, l2=1e-5)
        bias_reg = regularizers.L2(1e-5)
        activity_reg = regularizers.L2(1e-6)

    if hidden_dim > 0:
        # hidden representation of input
        hidden = Dense(hidden_dim, activation='relu', 
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=activity_reg)(input_img)
        # encoded representation of input
        encoded = Dense(encoding_dim, activation='relu',
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=activity_reg)(hidden)
    else:
        # encoded representation of input
        encoded = Dense(encoding_dim, activation='relu', 
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=activity_reg)(input_img)

    if hidden_dim > 0:
        # hidden representation of code 
        hidden = Dense(hidden_dim, activation='relu',
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=activity_reg)(encoded)
        # decoded representation of code 
        decoded = Dense(in_size, activation='sigmoid',
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=activity_reg)(hidden)
    else:
        # decoded representation of code 
        decoded = Dense(in_size, activation='sigmoid', 
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=activity_reg)(encoded)
    
    # Model which take input image and shows decoded images
    autoencoder = Model(input_img, decoded)


    # This model shows encoded images
    encoder = Model(input_img, encoded)
    # Creating a decoder model
    encoded_input = Input(shape=(encoding_dim,))
    # last layer of the autoencoder model

    if hidden_dim > 0:
        hidden_layer = autoencoder.layers[-2]
        decoder_layer = autoencoder.layers[-1]#from encoding to end
        decoder = Model(encoded_input, decoder_layer(hidden_layer(encoded_input)))
    else:
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))
    
    # decoder model
    #decoder = Model(encoded, decoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder, encoder, decoder



def extract_encoder_decoder(autoencoder):   # The autoencoder have to be symetric

    layer_len = len(autoencoder.layers)
    coding_size = layer_len//2
    print("Coding depth:", coding_size)

    input_img = Input(shape=(autoencoder.input_shape[1],), name=f'input{autoencoder.input_shape[1]}')
    input_lyr = autoencoder.layers[0](input_img)

    if coding_size == 1:
        encoded_lyr = autoencoder.layers[1](input_lyr)
        encoding_dim = encoded_lyr.shape[1]
    elif coding_size == 2:
        encoded_lyr = autoencoder.layers[1](input_lyr)
        encoded_lyr = autoencoder.layers[2](encoded_lyr)
        encoding_dim = encoded_lyr.shape[1]
    else: print("Unknown Size")

    encoded_input = Input(shape=(encoding_dim,))
    
    if coding_size == 1:
        decoder_lyr = autoencoder.layers[-1](encoded_input) # [2]
    elif coding_size == 2:
        decoder_lyr = autoencoder.layers[-2](encoded_input) # [2]
        decoder_lyr = autoencoder.layers[-1](decoder_lyr)   # [3]
    else: print("Unknown Size")

    encoder = Model(input_img, encoded_lyr)
    decoder = Model(encoded_input, decoder_lyr)

    # x_img = x.reshape(1,np.prod(x.shape))
    # encoded_val = encoder(x_img)
    # decoded_val = decoder(encoded_val)
    # decoded_img = decoded_val.numpy().reshape(x.shape)

    # autoencoder.layers[-1].input_shape # (None,32)
    # autoencoder.layers[-1].output_shape # (None, 1476)

    return encoder, decoder
    
import time
def Save(model, name="temp", model_type = "autoencoder", write_time=True):
    if write_time:
        model.save(f'logs/models/{model_type}_{time.strftime("[%j-%H-%M-%S]")}_{name}')
    else:
        model.save(f'logs/models/{model_type}_{name}')


def Load(name = "temp", time_str=None, model_type = "autoencoder", width=36):
    if time_str:
        loaded_model = load_model(f'logs/models/{model_type}_{time_str}_{name}', compile=False)
    else:
        loaded_model = load_model(f'logs/models/{model_type}_{name}', compile=False)
    loaded_model.compile(optimizer='adam', loss='binary_crossentropy') 
    img_shape = (int(loaded_model.input_shape[1]/width), width)
    return loaded_model, img_shape

class EarlyStoppingAtMinLoss(Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def fit(model, train_ds, val_ds, epoch=500, batch=256):
    model.fit(train_ds,
        epochs=epoch,
        batch_size=batch,
        validation_data=val_ds,
        callbacks=[EarlyStoppingAtMinLoss(patience=32)])
    return model
