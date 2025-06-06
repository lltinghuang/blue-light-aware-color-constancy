import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from guided_filter.core.filter import GuidedFilter

# hyper param
eta = 36
lmda = 10

def f(path,name,savepath,img2 = None,):
    img = cv2.imread(os.path.join(path,name))
    h, w = img.shape[0], img.shape[1]
    b, g, r = cv2.split(img)
    # global adaptation 
    # Y =  0.299 * r + 0.587 * g + 0.114 * b
    # U = -0.147 * r - 0.289 * g + 0.436 *b
    # V = 0.615*r  - 0.515 * g - 0.10 * b
    # r = Y + 1.14  * V
    # g = Y - 0.39 * U - 0.58 * V
    # b = Y + 2.03 * U
    
    '''
    We ignore the global tone mapping
    '''
    l = 0.299 * r + 0.587 * g + 0.114 * b
    l = l.astype(np.float32)
    
    l_log = l / 255
    print(f"max : {np.max(l_log)} min : {np.min(l_log)}" )
    # local adaptation
    GF = GuidedFilter(l_log, 10, 0.01)
    Hg = GF.filter(l_log)

    alpha = 1 + eta * l_log / np.max(l_log)  

    Lgaver = np.exp(np.sum(np.log(0.001 + l_log)) / (h * w))
    
    beta = lmda * Lgaver
    Lout = alpha * np.log(l_log / Hg + beta)
    
    # normalixed 
    Lout = cv2.normalize(Lout, None ,0, 255, cv2.NORM_MINMAX)
    
    eps = 1e-6  # or 1e-3 depending on your image scale
    gain = Lout / (l + eps)
        
    print(f"gain max: {np.max(gain)} gain min: {np.min(gain)}")
    
    gain[gain <= 0] = 0
    '''
    Template approach // suppose we have the maximum rgb in some value
    '''
    # Target output ranges
    r_target_max = 255.0
    g_target_max = 168.0
    b_target_max = 87.0

    # Normalize each channel to [0, 1] before applying custom scaling
    r_norm = np.clip(r * gain, 0, None)
    g_norm = np.clip(g * gain, 0, None)
    b_norm = np.clip(b * gain, 0, None)

    r_norm /= np.max(r_norm) + eps
    g_norm /= np.max(g_norm) + eps
    b_norm /= np.max(b_norm) + eps

    # Scale to desired fixed range
    r_out = r_norm * r_target_max
    g_out = g_norm * g_target_max
    b_out = b_norm * b_target_max

    # Merge channels and convert to 8-bit
    merged = cv2.merge([
        np.clip(b_out, 0, 255),
        np.clip(g_out, 0, 255),
        np.clip(r_out, 0, 255)
    ])
    
    # merged is a (H, W, 3) image in BGR format
    print(f"merged b max : {np.max(merged[:, :, 0])} b min : {np.min(merged[:, :, 0])}")
    print(f"merged g max : {np.max(merged[:, :, 1])} g min : {np.min(merged[:, :, 1])}")
    print(f"merged r max : {np.max(merged[:, :, 2])} r min : {np.min(merged[:, :, 2])}")
    
    out = cv2.convertScaleAbs(merged)

    if os.path.isdir(savepath) is not True:
        os.mkdir(savepath)
    cv2.imwrite(os.path.join(savepath,name), out)

def folder_test(path,savepath):
    names = os.listdir(path)
    for i,name in enumerate(names):
        f(path,name,savepath)

if __name__ == '__main__':
    ## Usage make the wanted folder like
    '''
    support batch handling
    ''' 
    ## ./ltm.py
    ## ./test_folder/target.jpg (png) 
    ## ./test_out_folder/
    imgs_path = 'test_folder'
    save_path = 'test_out_folder'
    folder_test(imgs_path,save_path)
