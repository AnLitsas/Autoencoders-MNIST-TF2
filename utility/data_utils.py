from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.datasets import mnist
import numpy as np
import random 
import scipy
import cv2
import os

class data_utils():
    def __init__(self):
        #Define the folder that stores the results
        self.outpath = "./Results"
        #Paths of folders that need to be created and save the results based on AE architecture
        self.outpath_ae = self.outpath+"/ae"
        self.outpath_batchnorm = self.outpath+"/batchnorm"
        self.outpath_fft = self.outpath+"/fft"
        self.outpath_dropout = self.outpath +"/dropout"
        self.outpaths = {
                        "ae": self.outpath_ae,
                        "batchnorm": self.outpath_batchnorm,
                        "fft": self.outpath_fft,
                        "dropout": self.outpath_dropout}

        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
     

    def create_path(self, ae_case, img_dim_case):
        try:
            to_path = self.outpaths[ae_case]
        except KeyError as e:
            print("False AE value is given! \n Check main.py for cases")
            print(e)
        if not os.path.isdir(to_path):
            os.mkdir(to_path)
        results_folder = to_path+"/Dim_"+str(img_dim_case)
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
        return(results_folder)

    
    def load_dt(self, img_dim_case):
        """
        Load your data: 
        """
        (X_train, _), (X_test, _) = mnist.load_data()
        
        # Resize images to img_dim_case x img_dim_case
        X_train = np.array([cv2.resize(img, (img_dim_case, img_dim_case)) for img in X_train])
        X_test = np.array([cv2.resize(img, (img_dim_case, img_dim_case)) for img in X_test])
        
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        X_train = np.reshape(X_train, (len(X_train), img_dim_case, img_dim_case, 1))
        X_test = np.reshape(X_test, (len(X_test), img_dim_case, img_dim_case, 1))

        X_val = X_test[:5000]
        X_test = X_test[5000:]

        # For visual inspection, you can select a few random samples
        visuals = np.random.choice(X_test.shape[0], 8)
        self.Xtest_visual = X_test[visuals]
        self.Xtrain_visual = X_train[visuals]

        return X_train, X_test, X_val, self.Xtest_visual, self.Xtrain_visual

    def absolute_spectrum(self):
        self.X_train = abs(scipy.fft.fft2(self.X_train)) #scipy.fft.fftshift(abs(scipy.fft.fft2(train)))
        self.X_test = abs(scipy.fft.fft2(self.X_test))
        self.X_val = abs(scipy.fft.fft2(self.X_val))
        self.Xtest_visual = abs(scipy.fft.fft2(self.Xtest_visual))
        self.Xtrain_visual = abs(scipy.fft.fft2(self.Xtrain_visual))

        self.X_train = self.X_train/self.x_train.max()
        self.X_test = self.X_test/self.X_test.max()
        self.X_val = self.X_val/self.X_val.max()
        self.Xtest_visual = self.Xtest_visual/self.Xtest_visual.max()
        self.Xtrain_visual = self.Xtrain_visual/self.Xtrain_visual.max()
        return(self.X_train, self.X_test, self.X_val, self.Xtest_visual, self.Xtrain_visual)
