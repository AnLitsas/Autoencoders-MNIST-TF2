import numpy as np
import tensorflow as tf
import sys
from architecture.AE_architectures import *
from utility.utils import *
from utility.data_utils import *
from tensorflow.python.keras.utils.vis_utils import plot_model


#change:for inbound_layer in node.inbound_layers:
    # to for inbound_layer in nest.flatten(node.inbound_layers): 
    #(Keras.utils.vis_utils.plot_model)
        
        
#Run on GPU
#gpus=tf.config.list_physical_devices('GPU')
#    tf.config.experimental.set_memory_growth(gpus[0], True)



#DEFINE IMAGE SIZES and AENs to build and fit 

dt = data_utils()
image_sizes=[64, 128, 256]
cases = ["ae", "batchnorm", "dropout", "fft"] # "fft" last or change line 49


##############################################################
#Outer LOOP:
#        Run every AEN by keeping the same image dimensions 
##############################################################
for im_size in image_sizes:
    (train, test, val, vis_test, vis_train) = dt.load_dt(im_size)
    
    '''
    Initialize a dict of test losses for each AE case and image dim 
    '''
    test_losses = {i:None for i in image_sizes}
    for j in image_sizes: test_losses[j] = {k: None for k in cases }
    
    ######################################
    #Inner LOOP:
    #       Build/fit/evaluate the models 
    #######################################
    for ind_c, case in enumerate(cases):
        results_folder = dt.create_path(case, im_size)
        
        #Get the absolute spectrum in case of fft
        if case=="fft": (train, test, val, vis_test, vis_train) = dt.absolute_spectrum()

        #Set the configurations

        configuration  =  set_configurations(dropout = True) if case == "dropout" else set_configurations(dropout = False)
        n_conf = len(configuration)

        ###############################################
        #2nd Inner LOOP:
        #    Build/fit/evaluate for every configuration
        ###############################################
        for i in range(len(configuration)):

            #create folder to store the results based on configuration
            new_folder = results_folder+"/"+str(configuration[i])
            if not os.path.isdir(new_folder):
                os.mkdir(new_folder)

            if case == "dropout":
                layers_size, latent_space, epochs, lr, batch_sz, sparc_strength, dropout_perc = configuration[i]
                test_losses[im_size][case] = {'config': [latent_space, lr, sparc_strength, dropout_perc], 'loss': None}
                AE = AE_architectures(im_size, latent_space, lr, layers_size, sparc_strength, case, dropout_perc, gen_n_nodes=0)

                encoder_input, encoded = AE.drop_encoder()
                reconstruction = AE.drop_decoder(encoded)
            else:
                layers_size, latent_space, epochs, lr, batch_sz, sparc_strength = configuration[i]
                test_losses[im_size][case] = {'config': [latent_space, lr, sparc_strength], 'loss': None}
                AE = AE_architectures(im_size, latent_space, lr, layers_size, sparc_strength, case, None, gen_n_nodes=0)

                if case == "ae":
                    encoder_input, encoded = AE.encoder()
                    reconstruction = AE.decoder(encoded)

                elif case == "batchnorm":
                    encoder_input, encoded = AE.batch_encoder()
                    reconstruction = AE.batch_decoder(encoded)

                elif case == "fft":
                    encoder_input, encoded = AE.encoder()
                    reconstruction = AE.decoder(encoded)
            ##################
            #Build Model
            ##################
            model = AE.model_build(encoder_input, reconstruction)
            print(model.summary())
            plot_model(model, to_file = new_folder+'/ae_plot.png', show_shapes=True, show_layer_names=True)

            ##########################
            #TRAIN MODEL 
            ##########################
            print("Training {} model of {}x{} images".format(case, im_size, im_size))
            if case == "dropout":
                print("Latent code of size {}, learning rate {}, sparcity strength of {} and dropout percentage of {}%".format(latent_space, lr, sparc_strength, dropout_perc*100))
            else:
                print("Latent code of size {}, learning rate {} and sparcity strength of {}".format(latent_space, lr, sparc_strength))
            [model, history] = AE.model_fit(model, train, val, batch_sz, epochs)
            ########################
            #Save history and model
            #load history: history = np.load('my_history.npy',allow_pickle='TRUE').item()
            ########################
            np.save(new_folder+'/my_history.npy',history.history)
            AE.model_save(model, new_folder)
            #############################
            #Evaluate test and train sets
            #############################
            preds_test = AE.model_evaluate(model, test)
            test_loss = ((preds_test-test)**2).mean()
            test_losses[im_size][case]['loss'] = test_loss

            preds_train = AE.model_evaluate(model, train)
            train_loss = ((preds_train - train)**2).mean()
            save_results(vis_test, vis_train, history, new_folder, model, im_size, test_loss, train_loss)

    print("Images of size {} x {}".format(im_size, im_size))
    for ind, i in enumerate(test_losses[im_size]):
        print("AE type of: {} ".format(i) )
        j = test_losses[im_size][i]['config']
        #for ind_j, j in enumerate(test_losses[im_size][i]['config']):
        if i == "dropout":
            print("Config: latent code of size {}, learning rate of {}, sparcity strength of {}, dropout percentage of {} \nwith test loss: {}".format(j[0], j[1], j[2], j[3], test_losses[im_size][i]['loss']))
        else:
            print("Config: latent code of size {}, learning rate of {}, sparcity strength of {} \nwith test loss: {}".format(j[0], j[1], j[2], test_losses[im_size][i]['loss']))

