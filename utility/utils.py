import numpy as np 
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import itertools
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img


def save_results(Xtest_visual, Xtrain_visual, history, new_folder, autoencoder, input_size, test_loss, train_loss):
    plt.plot(history.history['loss'], label="training loss")
    plt.plot(history.history['val_loss'], label="validation loss")
    plt.title("mse loss over training")
    plt.xlabel('mse loss')
    plt.ylabel('epochs')
    plt.legend()
    plt.savefig(new_folder+"/loss.png")
    plt.close()
    

    #Predictions Test set
    preds = autoencoder.predict(Xtest_visual)
    #Get the number of images 
    n=Xtest_visual.shape[0]
    #Set the figure size
    fig = plt.figure(figsize=(8, 2))
    fig.suptitle("Test Set: {:.5f} loss ".format(test_loss))
    for i in range(n):
        #Original images
        ax = plt.subplot(2, n, i+1)
        plt.imshow(Xtest_visual[i].reshape(input_size, input_size),cmap=plt.cm.gray, interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #Reconstructed images
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(preds[i].reshape(input_size, input_size), cmap=plt.cm.gray, interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    #plt.title("Original and reconstructed images")
    plt.savefig(new_folder+"/test_rec.png")
    plt.close()

    #Predictions Train set
    preds = autoencoder.predict(Xtrain_visual)
    #Get the number of images 
    n=Xtrain_visual.shape[0]
    #Set the figure size
    #plt.figure(figsize=(8, 2))
    
    fig = plt.figure(figsize=(8, 2))
    fig.suptitle("Train Set: {:.5f} loss ".format(train_loss))
    for i in range(n):
        #Original images
        ax = plt.subplot(2, n, i+1)
        plt.imshow(Xtrain_visual[i].reshape(input_size, input_size),cmap=plt.cm.gray, interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #Reconstructed images
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(preds[i].reshape(input_size, input_size), cmap=plt.cm.gray, interpolation='nearest')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    #plt.title("Original and reconstructed images")
    plt.savefig(new_folder+"/train_rec.png")
    plt.close()

def set_configurations(dropout):
    layers = [[8, 16, 32, 64]] 
    latent_sizes = [10, 20, 50, 100, 150] #, 20, 50, 100, 150] 
    epochs = [700]
    learning_rates =[0.00001] 
    batch_sizes =[64]
    sparsity_strengths=[0, 10e-6, 10e-7, 10e-8]
    if dropout==True: 
        drop_percs=[0.1, 0.2, 0.3, 0.4]
        config = [layers, latent_sizes, epochs, learning_rates, batch_sizes, sparsity_strengths, drop_percs]
        configuration= [ c for c in itertools.product(*config)]
        return(configuration)
    else:
        config = [layers, latent_sizes, epochs, learning_rates, batch_sizes, sparsity_strengths]
        configuration= [ c for c in itertools.product(*config)]
        return(configuration)

