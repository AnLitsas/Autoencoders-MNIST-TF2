from tensorflow import keras
import tensorflow as tf
from architecture.layers import *

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
        #self.select_ae ={"ae": {'encoder': encoder(), 'decoder': decoder(self.encoded) }}

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
    '''
    #for 64x64 images
    def discriminator(self):
        discriminator_input = keras.Input(shape=(self.w, self.h, self.c))
        #64x64x8
        conv_1  = first_conv(discriminator_input, self.l1, self.w, self.h, self.c, self.sparsity_str)
        if self.AE_case=="dropout": 
            drop_1 =dropout(conv_1, self.drop_perc)
            conv_2 = conv(drop_1, self.l2, self.sparsity_str)
            drop_2 =dropout(conv_2, self.drop_perc)
            conv_2 = drop_2 
        elif self.AE_case == "batchnorm":
            drop_1 =batchnormalization(conv_1)
            conv_2 = conv(drop_1, self.l2, self.sparsity_str)
            drop_2 =batchnormalization(conv_2)
            conv_2 = drop_2 
        else:
            #64x64x16
            conv_2 = conv(conv_1, 16, self.sparsity_str)
        #65536
        unfold = flatten(conv_2)
        output = dense(unfold, 1, 'sigmoid')
        model = keras.Model(discriminator_input, output, name="discriminator")

        optim = keras.optimizers.Adam(lr=self.lr, decay=1e-6)
        model.compile(optim, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return (model) 

    def generator(self):
        generator_input = keras.Input(shape = (self.latent_size))
        dense_1 = dense(generator_input, 8*8*self.gen_n_nodes, 'relu')
        #8x 8 x gen_n_nodes
        reshape_1 = reshape(dense_1, 8, 8, self.gen_n_nodes)
        #upsample to 16 x 16 x gen_n_nodes
        unpool_1 = unpool(reshape_1)
        deconv_1 = deconv(unpool_1, self.gen_n_nodes, self.sparsity_str)
        #upsample to 32 x 32 x gen_n_nodes
        unpool_2 = unpool(deconv_1)
        deconv_2 = deconv(unpool_2, self.gen_n_nodes, self.sparsity_str)
        #upsample to 64 x 64 x gen_n_nodes
        unpool_3 = unpool(deconv_2)
        output = last_deconv(unpool_3, 1 , self.sparsity_str)

        model = keras.Model(generator_input, output, name="generator")

        #optim = keras.optimizers.Adam(lr=self.lr, decay = 1e-6)
        #model.compile(optim, loss = '')
        return(model)


    def gan(self, d_model, g_model):
        d_model.trainable = False
       
        gan_model = keras.models.Sequential()
        gan_model.add(g_model)
        
        gan_model.add(d_model)
        # compile model
        opt = keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        gan_model.compile(loss='binary_crossentropy', optimizer=opt)
        return(gan_model)
    '''
    def model_fit(self, autoencoder, X_train, X_val, batch_sz, epochs):
        print("Fitting..")
    
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = autoencoder.fit(X_train, X_train, epochs=epochs,
                                  batch_size=batch_sz, validation_data = (X_val, X_val), verbose=0,
                                  callbacks=[callback] )

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
        autoencoder.compile(optim, loss='mse', metrics=['accuracy'])
        return(autoencoder)

