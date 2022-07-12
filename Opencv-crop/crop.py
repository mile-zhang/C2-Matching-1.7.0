# -*- coding: utf-8 -*-
import cv2
import os 

path = 'C:/Users/re58g/桌面/Opencv-crop' # image local path 
files = os.listdir(path)
img_file = './crop' # crop_image save path (must use relative path not absolute path)

# Crop: x,y -> Left begin (x,y) coordinate。 w:width, h:height
def Crop(x,y,w,h):
    cropped = img[y:y+h, x:x+w]
    return cropped

def save_crop_img(file,crop_img):
    if not os.path.exists(img_file):
        os.mkdir(img_file)
    print (img_file+ "/" + file)
    cv2.imwrite(img_file+ "/" + file, crop_img)
    cv2.waitKey(0)
    return  

for file in files:
    if file.endswith('.png'):
        img = cv2.imread(file)
        print(img.shape)
        print(type(img))
        crop_img = Crop(100,200,200,150) # 001 -> 250,100,200,150 
        save_crop_img (file,crop_img)
        # cv2.imshow('image', crop_img)
        # cv2.imshow('cropped', crop_img)s