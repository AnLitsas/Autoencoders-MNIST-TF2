from tensorflow.python.keras.utils.vis_utils import plot_model
from architecture.AE_architectures import *
from utility.generategif import GenerateGIF
from utility.data_utils import *
from utility.utils import *
import tensorflow as tf
import numpy as np
import imageio
import sys


# Initialize data utility and define image sizes and autoencoder types
dt = data_utils()
image_sizes=[64] # [64, 128, 256] ---> Run all the different image sizes experiments
cases = ["ae"] # ["ae", "batchnorm", "dropout", "fft"] 

# Iterate through each image size to build, fit, and evaluate autoencoders
for im_size in image_sizes:
    (train, test, val, vis_test, vis_train) = dt.load_dt(im_size)
    test_losses = {i:None for i in image_sizes}
    for j in image_sizes: test_losses[j] = {k: None for k in cases }
    
    # Iterate through each autoencoder type
    for ind_c, case in enumerate(cases):
        results_folder = dt.create_path(case, im_size)
        
        # Apply Fast Fourier Transform if the case is 'fft'
        if case=="fft": (train, test, val, vis_test, vis_train) = dt.absolute_spectrum()

        # Configure hyperparameters
        configuration  =  set_configurations(dropout = True) if case == "dropout" else set_configurations(dropout = False)
        n_conf = len(configuration)

        # Iterate through each hyperparameter configuration
        for i in range(len(configuration)):
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
            
            
            # Build and summarize the model
            model = AE.model_build(encoder_input, reconstruction)
            print(model.summary())
            #plot_model(model, to_file = new_folder+'/ae_plot.png', show_shapes=True, show_layer_names=True)

            # Initialize the custom callback
            generate_gif_callback = GenerateGIF(vis_test, im_size)
            
            # Train the model
            print("Training {} model of {}x{} images".format(case, im_size, im_size))
            if case == "dropout":
                print("Latent code of size {}, learning rate {}, sparcity strength of {} and dropout percentage of {}%".format(latent_space, lr, sparc_strength, dropout_perc*100))
            else:
                print("Latent code of size {}, learning rate {} and sparcity strength of {}".format(latent_space, lr, sparc_strength))
            #[model, history] = AE.model_fit(model, train, val, batch_sz, epochs)
            [model, history] = AE.model_fit(model, train, val, batch_sz, epochs, callbacks=[generate_gif_callback])

            
            # Save training history and model
            np.save(new_folder+'/my_history.npy',history.history)
            AE.model_save(model, new_folder)
            
            # Evaluate the model on test and train sets
            preds_test = AE.model_evaluate(model, test)
            test_loss = ((preds_test-test)**2).mean()
            test_losses[im_size][case]['loss'] = test_loss

            preds_train = AE.model_evaluate(model, train)
            train_loss = ((preds_train - train)**2).mean()
            save_results(vis_test, vis_train, history, new_folder, model, im_size, test_loss, train_loss)

        # Create a GIF from frames
        imageio.mimsave(f'{new_folder}/training_progress.gif', generate_gif_callback.frames, duration=0.08, loop=0)

    # Display test loss for each configuration and autoencoder type
    print("Images of size {} x {}".format(im_size, im_size))
    for ind, i in enumerate(test_losses[im_size]):
        print("AE type of: {} ".format(i) )
        j = test_losses[im_size][i]['config']
        if i == "dropout":
            print("Config: latent code of size {}, learning rate of {}, sparcity strength of {}, dropout percentage of {} \nwith test loss: {}".format(j[0], j[1], j[2], j[3], test_losses[im_size][i]['loss']))
        else:
            print("Config: latent code of size {}, learning rate of {}, sparcity strength of {} \nwith test loss: {}".format(j[0], j[1], j[2], test_losses[im_size][i]['loss']))

