import numpy as np
import pickle
import cv2
import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
from scipy.io import wavfile
from detect_peaks import detect_peaks
import os
import os.path
import gc
import re
import tgt
import csv
import datetime
import scipy
import pickle
import random
random.seed(17)
import skimage
from subprocess import call, check_output, run
from scipy.io.wavfile import read, write


# sample from Csaba
import WaveGlow_functions

import keras
from keras import regularizers
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, InputLayer, Dropout
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
# additional requirement: SPTK 3.8 or above in PATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def read_ult(filename, NumVectors, PixPerVector):
    # read binary file
    ult_data = np.fromfile(filename, dtype='uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data


# parameters of ultrasound images, from .param file
n_lines = 64
n_pixels = 842

# reduce ultrasound image resolution
n_pixels_reduced = 128

# TODO: modify this according to your data path
dir_base = '/data/UltraSuite_TaL/TaL80/core/'
# speakers = ['01fi', '02fe', '03mn', '04me', '05ms', '06fe', '07me', '08me', '09fe', '10me']
speakers = ['01fi', '02fe', '03mn', '04me']
# speakers = ['aze_spkr01']

for speaker in speakers:

    # collect all possible ult files
    ult_files_all = []
    dir_data = dir_base + speaker + '/'
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            # collect _aud and _xaud files
            # for i in range(int(154*0.7)):
            #     if file.endswith('.augult'):
            #         ult_files_all += [dir_data + file[:-7]]
            # if file.endswith('.augult'):
            #     ult_files_all += [dir_data + file[:-7]]
            if file.endswith('.ult'):
                ult_files_all += [dir_data + file[:-4]]
    # randomize the order of files
    random.shuffle(ult_files_all)



    def delete_files(directory:str):
        for filename in os.listdir(directory):
            if filename.endswith('_cM.augult'):
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    # print(f"{filename} has been removed.")
                except Exception as e:
                    print(f"Error occured while removing {filename} : {e}")
        print("All cM augmented files have been removed")
        for filename in os.listdir(directory):
            if filename.endswith('_iM.augult'):
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    # print(f"{filename} has been removed.")
                except Exception as e:
                    print(f"Error occured while removing {filename} : {e}")

        print("All iM augmented files have been removed")


    def cMaskingU(file): #consecutive time masking for ult data randomly starting point in both dimensions
            b = read_ult(file, n_lines,n_pixels)
            data = b.copy()
            a = np.random.randint(low = 0, high=data.shape[0], size=None, dtype=int)
            if a+10>data.shape[0]:
                a = np.random.randint(low = 0, high=data.shape[0], size=None, dtype=int)
            b = a + 10
            array = data[a:b]
            for j in range(array.shape[0]):
              st = random.randint(50,150)
              nd = st + 50
                # main = array[st:nd]
              for j in range(array.shape[0]):
                    # print(array.shape[0])
                    line = array[j]
                    for k in range(line.shape[0]):
                      main = line[k]
                      main[st:nd] = 0
            return data

    def iMaskingU(file): #intermittent time masking for ult data randomly starting point in both dimensions
            b = read_ult(file, n_lines,n_pixels)
            data = b.copy()
            a = random.randint(0,data.shape[0])
            startlist = [a]
            endList = [a+10]
            for i in range(0,5):
                n = random.randint(0,data.shape[0])
                if startlist[i-1] in range(n,n+10,1) and n+10>data.shape[0]:
                  n = random.randint(0,data.shape[0])
                  startlist[i-1] = n
                  endList[i-1] = n+10
                endList.append(n+10)
                startlist.append(n) 
            for i in range(len(startlist)):
                a = startlist[i]
                b = endList[i]
                array = data[a:b]
                st = random.randint(50,150)
                nd = st + 50
                # main = array[st:nd]
                for j in range(array.shape[0]):
                    # print(array.shape[0])
                    line = array[j]
                    for k in range(line.shape[0]):
                      main = line[k]
                      main[st:nd] = 0
                      # print(main.shape)

            return data


    def edge_En(file):
        # Define the kernel size
        ksize = 15
        b = read_ult(file, n_lines,n_pixels)
        # Apply the edge-enhancing filter to each slice of the image
        filtered_b = np.zeros_like(b)
        for i in range(b.shape[0]):
            blurred = cv2.GaussianBlur(b[i], (ksize, ksize), 0)
            filtered_b[i] = cv2.addWeighted(b[i], 1.5, blurred, -0.5, 0)
        return filtered_b

    def dMaskingU(file): #consecutive time masking for ult data
            b = read_ult(file, n_lines,n_pixels)
            data = b.copy()
            a = random.randint(50,150)
            startlist = [a]
            endList = [a+5]
            for i in range(0,2):
                n = random.randint(50,150)
                if startlist[i-1] in range(n,n+5,1) and n+5>150:
                  n = random.randint(50,150)
                  startlist[i-1] = n
                  endList[i-1] = n+5
                endList.append(n+5)
                startlist.append(n)
            for i in range(len(startlist)):
                s = startlist[i]
                e = endList[i]
                for j in range(data.shape[0]):
                    array = data[j]
                    for k in range(array.shape[0]):
                        line = array[k]
                        line[s:e] = 0
            return data

    def sNoiseI_U(file): #sinusoidal noise injection for ult data
        b = read_ult(file, n_lines,n_pixels)
        data = b.copy()
        for i in range(data.shape[0]):
          array = data[i]
          for j in range(array.shape[0]):
            a = 0.02
            mean_value = np.mean(array[j])
            f = 40
            noise = a * mean_value * np.sin(2 * np.pi * f * np.arange(array.shape[1])/len(array[j]))
            array[j] = array[j] + noise    
        return data

    def rScalingU(file): #random scaling for ult data
        b = read_ult(file, n_lines,n_pixels)
        data = b.copy()
        for i in range(data.shape[0]):
          array = data[i]
          for j in range(array.shape[0]):
            scale_factor = np.random.uniform(low=0.8, high=1.4)
            array[j] = array[j] * scale_factor
            # array[j] = np.clip(array[j], -1, 1)
        return data 

            
        

    def write_ult(filename, ult_data):
        # ensure ult_data is of type uint8
        ult_data = np.asarray(ult_data, dtype='uint8')
        # reshape data to flatten first dimension
        ult_data = np.reshape(ult_data, (-1,))
        # write data to binary file
        ult_data.tofile(filename)

    def saveDA():
        for basefile in ult_files_all:
            # data1 = cMaskingU(basefile + '.ult')
            # data2 = iMaskingU(basefile + '.ult')
            # data = sNoiseI_U(basefile + '.ult')
            # data = rScalingU(basefile + '.ult')
            data = dMaskingU(basefile + '.ult')
            # data3 = edge_En(basefile + '.ult')
            # data = cMaskingU_new(basefile + '.ult')
            # data = iMaskingU_new_file(basefile + '.ult')
            # data = merge_ult(basefile + '.ult', basefile + '.ult')
            # with open(basefile + '_ran_new.cult', 'wb') as f:
            #     pickle.dump(data, f)  
            write_ult(basefile + '_dM.augult',data)
           
        return 0


    # delete_files(dir_data)
    # changeULT()
    # print("ult format has been changed to pickle!")
    # saveDA()
    # print("DDR is done")
    # print("CTM is done")
    # print("ITM has done")
    # print("EE has done")
    # print("SNI has done")
    # print("RS has done")
    # print("Merging has done")

    