from tensorflow.keras import metrics
from architecture.layers import *
from tensorflow import keras
import tensorflow as tf


class AE_architectures():
    def __init__(self, shape, latent_size, lr, layers_size, sparsity_str, AE_case, drop_perc, gen_n_nodes):
        self.w = shape
        self.h = shape
        self.c = 1
        self.latent_size = latent_size
        self.lr = lr 
        self.l1 = layers_size[0]
        self.l2 = layers_size[1]
        self.l3 = layers_size[2]
        self.l4 = layers_size[3] 
        self.sparsity_str = sparsity_str
        self.AE_case = AE_case
        self.drop_perc = drop_perc
        self.gen_n_nodes = gen_n_nodes

    def encoder(self):
        encoder_input = keras.Input(shape=(self.w, self.h, self.c), name='img')
        conv_1 = first_conv(encoder_input, self.l1 , self.w, self.h, self.c, self.sparsity_str)
        max_pool_1 = maxpool(conv_1)

        conv_2 = conv(max_pool_1, self.l2, self.sparsity_str)
        max_pool_2 = maxpool(conv_2)

        conv_3 = conv(max_pool_2, self.l3, self.sparsity_str)
        max_pool_3 = maxpool(conv_3)

        conv_4 = conv(max_pool_3, self.l4, self.sparsity_str)
        max_pool_4 = maxpool(conv_4)

        global dim1, dim2, dim3, dim4
        (dim1, dim2, dim3, dim4) = max_pool_4.get_shape()

        """
        Latent Space
        """
        unfold = flatten(max_pool_4)
        self.encoded = dense(unfold, self.latent_size, 'relu')

        return (encoder_input, self.encoded)

    def decoder(self, encoded):
        decoded = dense(encoded, dim2*dim3*dim4, 'relu')
        fold = reshape(decoded, dim2, dim3, dim4)
     
        unpool_1 = unpool(fold)
        deconv_1 = deconv(unpool_1, self.l3, self.sparsity_str)

        unpool_2 = unpool(deconv_1)
        deconv_2 = deconv(unpool_2, self.l2, self.sparsity_str)

        unpool_3 = unpool(deconv_2)
        deconv_3 = deconv(unpool_3, self.l1, self.sparsity_str)

        unpool_4 = unpool(deconv_3)
        reconstruction = last_deconv(unpool_4, 1, self.sparsity_str)
        
        return(reconstruction)

    def batch_encoder(self):
        encoder_input = keras.Input(shape=(self.w, self.h, self.c), name='img')
        conv_1 = first_conv(encoder_input, self.l1 , self.w, self.h, self.c, self.sparsity_str)
        max_pool_1 = maxpool(conv_1)
        batch_1 = batchnormalization(max_pool_1)

        conv_2 = conv(batch_1, self.l2, self.sparsity_str)
        max_pool_2 = maxpool(conv_2)
        batch_2 = batchnormalization(max_pool_2)

        conv_3 = conv(batch_2, self.l3, self.sparsity_str)
        max_pool_3 = maxpool(conv_3)
        batch_3 = batchnormalization(max_pool_3)

        conv_4 = conv(batch_3, self.l4, self.sparsity_str)
        max_pool_4 = maxpool(conv_4)

        global dim1, dim2, dim3, dim4
        (dim1, dim2, dim3, dim4) = max_pool_4.get_shape()

        """
        Latent Space
        """
        unfold = flatten(max_pool_4)
        self.encoded = dense(unfold, self.latent_size, 'relu')

        return (encoder_input, self.encoded)

    def batch_decoder(self, encoded):
        decoded = dense(encoded, dim2*dim3*dim4, 'relu')
        fold = reshape(decoded, dim2, dim3, dim4)
     
        unpool_1 = unpool(fold) 
        deconv_1 = deconv(unpool_1, self.l3, self.sparsity_str)
        batch_1 =batchnormalization(deconv_1)

        unpool_2 = unpool(batch_1)
        deconv_2 = deconv(unpool_2, self.l2, self.sparsity_str)
        batch_2 =batchnormalization(deconv_2)

        unpool_3 = unpool(batch_2)
        deconv_3 = deconv(unpool_3, self.l1, self.sparsity_str)
        batch_3 =batchnormalization(deconv_3)

        unpool_4 = unpool(batch_3)
        reconstruction = last_deconv(unpool_4, 1, self.sparsity_str)
        
        return(reconstruction)

    def drop_encoder(self):
        encoder_input = keras.Input(shape=(self.w, self.h, self.c), name='img')
        conv_1 = first_conv(encoder_input, self.l1 , self.w, self.h, self.c, self.sparsity_str)
        max_pool_1 = maxpool(conv_1)
        drop_1 = dropout(max_pool_1, self.drop_perc)

        conv_2 = conv(drop_1, self.l2, self.sparsity_str)
        max_pool_2 = maxpool(conv_2)
        drop_2 = dropout(max_pool_2, self.drop_perc)

        conv_3 = conv(drop_2, self.l3, self.sparsity_str)
        max_pool_3 = maxpool(conv_3)
        drop_3 = dropout(max_pool_3, self.drop_perc)

        conv_4 = conv(drop_3, self.l4, self.sparsity_str)
        max_pool_4 = maxpool(conv_4)

        global dim1, dim2, dim3, dim4
        (dim1, dim2, dim3, dim4) = max_pool_4.get_shape()

        """
        Latent Space
        """
        unfold = flatten(max_pool_4)
        encoded = dense(unfold, self.latent_size, 'relu')

        return (encoder_input, encoded)

    def drop_decoder(self, encoded):
        decoded = dense(encoded, dim2*dim3*dim4, 'relu')
        fold = reshape(decoded, dim2, dim3, dim4)
     
        unpool_1 = unpool(fold)
        deconv_1 = deconv(unpool_1, self.l3, self.sparsity_str)
        drop_1 =dropout(deconv_1, self.drop_perc)

        unpool_2 = unpool(drop_1)
        deconv_2 = deconv(unpool_2, self.l2, self.sparsity_str)
        drop_2 =dropout(deconv_2, self.drop_perc)

        unpool_3 = unpool(drop_2)
        deconv_3 = deconv(unpool_3, self.l1, self.sparsity_str)
        drop_3 =dropout(deconv_3, self.drop_perc)

        unpool_4 = unpool(drop_3)
        reconstruction = last_deconv(unpool_4, 1, self.sparsity_str)
        
        return(reconstruction)
    
    def model_fit(self, autoencoder, X_train, X_val, batch_sz, epochs, callbacks=None):
        print("Fitting..")
        early_stopping_callback  = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        if callbacks:
            callbacks.append(early_stopping_callback )
        else:
            callbacks = [early_stopping_callback ]
            
        history = autoencoder.fit(X_train, X_train, epochs=epochs,
                                  batch_size=batch_sz, validation_data = (X_val, X_val), verbose=1,
                                  callbacks=callbacks )

        return(autoencoder, history)

    def model_save(self, autoencoder, new_folder):
        autoencoder.save(new_folder+"/model.h5")

    def model_evaluate(self, autoencoder, X_test):
        preds = autoencoder.predict(X_test)
        return(preds)

    def model_build(self, encoder_input, reconstruction):
        autoencoder = keras.Model(encoder_input, reconstruction)
        """
        Select optimizer
        """
        optim = keras.optimizers.Adam(lr=self.lr, decay=1e-6)
        """
        Compile the Autoencoder
        """
        autoencoder.compile(optim, loss='mse', metrics=[metrics.MeanAbsoluteError()])
        return(autoencoder)

