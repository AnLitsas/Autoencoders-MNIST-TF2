%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MT: Autencoder package - python 
	 Anastasis Litsas
	   August 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Package includes: 
1) main.py
2) AE_architectures.py
3) layers.py
4) utils.py
5) data_utils.py


1) 	As the main script, it contains information that can be changed relating the size of the images and cases of AE (normal, batchnormalization, dropout, fft images) it will train and evaluate. 
	First loop: runs for each size of images having been it been determined.
	Second loop: runs for each case of AE.
	Third loop: runs for each configuration. Configurations can be changed in "utils.py"
	
	Whilst the latter loop is running:
		i)	The "AE_architectures.py" script is being called to create an AE object with the chosen hyperparameters (configurations).
		ii)	Then, construct both encoder and decoder by calling the right functions (e.g. "AE_architectures.encoder()", "AE_architectures.decoder()" ) from "AE_architectures.py" and build the autoencoder 
			(i.e. "AE_architectures.model_build (encoder, decoder)")
		iii)	Finally, the model is trained, evaluated and saved.	

2)	Contains the class responsible to create the desired autoencoder (also a simple gan model).
	
	Functions: 
		i) __init__(self, shape, latent_size, lr, layers_size, sparsity_str, AE_case, drop_perc, gan_n_nodes), where "shape" is the size of images we want to train the autoencoder with, "latent_size" is the size of the encoded code, "lr" is the 			learning rate, "layers_size" is a list of 4 integers representing the number of nodes for each layer, "sparsity_str" and "drop_perc" is the sparsity strength and drop percentage respectively, "AE_case" is a string choosing the 			autoencoder variation 
		(original AE: "ae", use of batch normilization: "batchnorm", use of dropout: "dropout", use of image's power spectrum: "fft")
		ii) encoder(): simple encoder used in cases of "ae", "fft"
		iii) decoder(encoder_input): same as b) 
		iv)  batch_encoder()
		v) batch_decoder(encoder_input)
		vi) drop_encoder()
		vii) drop_decoder(encoder_input)
        viii) model_fit(autoencoder, X_train, X_val, batch_sz, epochs)
        ix) model_save(autoencoder, destination_path)
        x) model_evaluate(autoencoder, X_test)
        xi) model_build(encoder_input, decoder)
        
        *xii) discriminator()
        *xiii) generator()
        *xiv) gan(d_model, g_model)

        Normal encoder consist 4 pairs of conv & max_pool layers, while iv) and vi) consist 4 pairs of conv, max_pool and batchnormilization or dropout layers. 
        Each encoder outputs both encoder input (tensor) and latent code ( produced by last max_pool -> flatten -> dense layer), while decoders output the reconstruction.

        "xi" determines loss type (default mean squared error, can be changed manually)

3)  Every possible layer is defined as a function using keras
	
	Functions:
		i) first_conv(X, layer_size, width, height, channels, sparsity_str): ReLU activation function 
		ii) conv(X, layer_size, sparsity_strength): ReLU activation function
		iii) maxpool(X): "tensorflow.keras.layers.MaxPooling2D()(X)"
		iv) flatten(X)
		v) dense(X, latent_size, activation_function): activation function must be selected 
		vi) reshape(X, dim1, dim2, dim3)
		vii) unpool(X): "tensorflow.keras.layers.UpSampling2D()(X)"
		viii) deconv(X, layer_size, sparsity_strength): kernel size of 5, activation function ReLU
		ix) last_deconv(X, layer_size, sparsity_strength): kernel size of 5, activation function sigmoid
		x) batchnormalization(X)
		xi) dropout(X, drop_perc)

4)	Functions:
		i) save_results(Xtest_visual, Xtrain_visual, history, folder_path, model, input_size, test_loss, train_loss): creates figure of training and validation loss over epochs. Also, creates 2 figures of N reconstructed images 			(X_test/train_visual, N=X_test_visual.shape[0]).
		ii) set_configurations(dropout): grid search method for tuning.
			dropout = False or True 
			default options (need mannual change):
				a) layers = [[8, 16, 32, 64]]  (lists of number of channels needed in the conv/deconv layers)
				b) latent_sizes = [10, 20, 50, 100, 150]
				c) epochs = [700]
				d) learning_rates = [0.00001]
				e) batch_sizes = [64]
				f) sparsity_strengths=[0, 10e-6, 10e-7, 10e-8

5) Class handling data.
	
   Functions: 
   	i) data_utils.create_path(ae_case, img_size)
   	ii) data_utils.load_data(img_size): outputs X_train, X_test, X_val, Xtest_visual, Xtrain_visual
   		visual data determined by the list: visuals=[932,  1014,  6,  2969,  2032,  3124,  562,  1545]
   		where Xtest_visual = X_test[visuals,:,:,:] (same goes for Xtrain)
   	iii) absolute_spectrum(): outputs the power spectrum of X_train, X_test, X_val, Xtest_visual, Xtrain_visual. 
   	     i.e: abs(scipy.fft.fft2(X))/ max(abs(scipy.fft.fft2(self.X_test)))	 (normalized)		
