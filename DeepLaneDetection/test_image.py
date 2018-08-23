from AirSimClient import *
import numpy as np
import pandas as pd
import PIL
#from PIL import Image
from skimage.transform import resize
 
car_client = CarClient()
car_client.confirmConnection()
# car_client.enableApiControl(True)
car_controls = CarControls()
car_client.simSetSegmentationObjectID("[\w]*", 0, True);
car_client.simSetSegmentationObjectID("Landscape13", 2, True);
 
foo = []
 
 
 
def get_image(car_client):
        image_response = car_client.simGetImages([ImageRequest(0, AirSimImageType.Segmentation, False, False)])[0]
        image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
        image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
 
        #Remove alpha channel
        image_rgba = image_rgba[32:132, 0:255, 0:3]
        
        #return image_rgba.astype(float)
        return image_rgba
        
def get_image_2(car_client):
        image_response = car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
        image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
        image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
 
        #Remove alpha channel
        image_rgba = image_rgba[32:132, 0:255, 0:3]
        
        #image_pil = PIL.Image.fromarray(image_rgba).convert('L').resize((image_rgba.shape[1]//2, image_rgba.shape[0]), resample=PIL.Image.LANCZOS)
        #image_data = np.asarray(image_pil.getdata(), dtype=float) / 255.0
        
        #return image_rgba.astype(float)
        #return image_data
        return image_rgba
        
im = get_image(car_client)
ii = PIL.Image.fromarray(im)
ii.save('out.jpg')

im = get_image_2(car_client)
ii = PIL.Image.fromarray(im)
ii.save('out_scene.jpg')