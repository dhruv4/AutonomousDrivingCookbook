import numpy as np
import cv2
from scipy.misc import imresize
import PIL
from math import isclose

def atPos(curr, tar):
    return curr[b'position'] == tar[b'position']

def nearPos(curr, tar):
    
    return isclose(curr[b'position'][b'x_val'], tar[b'position'][b'x_val'], rel_tol=0.005) \
        and isclose(curr[b'position'][b'y_val'], tar[b'position'][b'y_val'], rel_tol=0.005) \
        and isclose(curr[b'position'][b'z_val'], tar[b'position'][b'z_val'], rel_tol=0.01)

def draw_lanes(model, image, label=np.array([]), predshape=(80, 160, 3), shape=(1080, 1632, 3)):
        
    # Resize images
    small_img = imresize(image, predshape)    
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]
            
    # Make prediction, if there is no given label
    if(label.shape != (0,)):
        lane_drawn = label
        
    else:

        prediction = model.predict(small_img)[0] * 255

        # Generate fake R & G color dimensions, stack with B
        blanks = np.zeros_like(prediction).astype(np.uint8)
        lane_drawn = np.dstack((blanks, blanks, prediction))
            
    # Re-size to match the original image
    lane_image = imresize(lane_drawn, shape)
        
    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
        
    return result

def predictImage(model, imagePath, shape=(100, 255, 3), USELABEL = False, label = ""):

    # test how the prediction responds to a given scene
    img = PIL.Image.open(imagePath)

    img = np.asarray(img)

    if USELABEL:

        label = imresize(PIL.Image.open(label), (80, 160, 3))

        label = np.asarray(label)

        res = draw_lanes(model, img, label=label, shape=(100, 255, 3))

        res = PIL.Image.fromarray(res)

    else:
        
        res = draw_lanes(model, img, shape=shape)

        res = PIL.Image.fromarray(res)
    
    return res