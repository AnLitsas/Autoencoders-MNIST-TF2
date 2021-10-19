from tensorflow.keras.preprocessing.image import img_to_array, load_img
import scipy
import random 
import numpy as np
import os

class data_utils():
    def __init__(self):
        #Define folders containing the datasets
        self.path_64 = "./data/dataset_64"
        self.path_128 = "./data/dataset_128"
        self.path_256 = "./data/dataset_256"
        self.paths = {
                        "64": self.path_64,
                        "128": self.path_128,
                        "256": self.path_256}
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
        #Paths of folders that need to be created and save the results based on image dimension
        #self.outpaths_dims = {
        #                      "ae": [self.outpath_ae+"/Dim_64", self.outpath_ae+"/Dim_128", self.outpath_ae+"/Dim_256" ],
        #                      "batchnorm": [self.outpath_batchnorm+"/Dim_64", self.outpath_batchnorm+"/Dim_128", self.outpath_batchnorm+"/Dim_256" ],  
        #                      "fft": [self.outpath_fft+"/Dim_64", self.outpath_fft+"/Dim_128", self.outpath_fft+"/Dim_256" ]}

        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
     

    def create_path(self, ae_case, img_dim_case):
        try:
            to_path = self.outpaths[ae_case]
            #result_folders = self.outpaths_dims[ae_case]
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
        try:
            file_path = self.paths[str(img_dim_case)]
        except KeyError as e:
            print("False image dimension is given...")
            print(e)

        train_dt=os.listdir(file_path+"/train")
        test_dt=os.listdir(file_path+"/test")
        val_dt=os.listdir(file_path+"/val")
        
        #Train data set
        train_file =file_path+"/train/"
        X_train_tmp=[]
        print("loading training set... ")
        
        for i in train_dt:
            selected_file = train_file+str(i)
            X_train_tmp.append(img_to_array(load_img(selected_file, color_mode="grayscale"))/255.0)
        
        X_train=random.sample(X_train_tmp, 10000)   
        self.X_train = np.array(X_train)


        #Test Data set
        test_file =file_path+"/test/"
        X_test=[]
        print("loading test set... ")
        for i in test_dt:
            selected_file = test_file+str(i)
            X_test.append(img_to_array(load_img(selected_file, color_mode="grayscale"))/255.0)
        self.X_test = np.array(X_test)

        #Validation Data set
        val_file = file_path+"/val/"
        X_val=[]
        print("loading validation set...")
        for i in val_dt:
            selected_file = val_file+str(i)
            X_val.append(img_to_array(load_img(selected_file, color_mode="grayscale"))/255.0)
        self.X_val = np.array(X_val)
        
        
        #8 pre-selected testing images for showing the results
        visuals=[932,  1014,  6,  2969,  2032,  3124,  562,  1545]
        self.Xtest_visual = self.X_test[visuals,:,:,:]
        self.Xtrain_visual = self.X_train[visuals,:,:,:]


        return(self.X_train, self.X_test, self.X_val, self.Xtest_visual, self.Xtrain_visual)

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
