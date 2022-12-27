from IPython.display import display, clear_output, update_display
import matplotlib.pyplot as plt
import numpy as np 
import PIL

def show_examples_of_autoencoder(ds, encoder, decoder, img_shape, wait=0.5):
    encoding_size = encoder.output_shape[1]
    for image in ds.take(1):
        x_test,_ = image
        encoded_img = encoder.predict(x_test)
        decoded_img = decoder.predict(encoded_img)
        for x, e, d in zip(x_test, encoded_img, decoded_img):
            #print(x.shape)
            x = np.reshape(x,(*img_shape,1))
            #plt.imshow(x)
            #print(x.shape)
            #print(d.shape)
            d = np.reshape(d,(*img_shape,1))
            #print(d.shape)
            #plt.imshow(d)
            #line = np.zeros((img_shape[0],1,1))
            #plt.imshow(np.hstack((x,line,d)))
            #plt.imshow(np.hstack((x,d)))
            e = np.reshape(e, (encoding_size//4,4,1))
            fig, ax = plt.subplots(1,3, gridspec_kw={'width_ratios': [3, 1, 3]})
            fig.set_size_inches(8, 4)
            
            ax[0].imshow(x, interpolation='nearest')
            ax[0].set_title("Original")
            ax[1].imshow(e, interpolation='nearest')
            ax[1].set_title("Encoded")
            ax[2].imshow(d, interpolation='nearest')
            ax[2].set_title("Decoded")
            fig.tight_layout()
            plt.pause(wait)
    plt.show()


ENABLE_DISPLAY = None
DISPLAYED_ONCE = False
def enable_display(id="render"):
    global DISPLAYED_ONCE, ENABLE_DISPLAY
    DISPLAYED_ONCE = False
    ENABLE_DISPLAY = id
def try_displaying(obs):
    global ENABLE_DISPLAY, DISPLAYED_ONCE
    if ENABLE_DISPLAY != None:
        obs_img = PIL.Image.fromarray(obs)
        obs_img = obs_img.resize((200,200))
        if DISPLAYED_ONCE:
            #clear_output(wait=False)
            #obs_img.save(f"images/test{i}.png")
            update_display(obs_img, display_id=ENABLE_DISPLAY)
        else:
            display(obs_img, display_id=ENABLE_DISPLAY)
            DISPLAYED_ONCE = True