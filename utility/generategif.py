from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import imageio
import io


class GenerateGIF(Callback):
    """
    Class: GenerateGIF
    
    Description:
    The GenerateGIF class is a custom Keras Callback designed to generate GIF frames 
    at the end of each training epoch. It captures the original and predicted images 
    from the test set and saves them as frames to be later compiled into a GIF.
    
    Inputs:
    - vis_test: A NumPy array containing the test set images that you want to visualize.
    - im_size: An integer specifying the dimensions of the images (assuming square images).
    - frames: A list to store the frames that will be used to create the GIF.
    """
    def __init__(self, vis_test, im_size):
        """
        Method: __init__(self, vis_test, im_size, frames)
        
        Description:
        The constructor method initializes the class attributes based on the provided inputs.
        
        Inputs:
        - vis_test: Test set images for visualization.
        - im_size: Dimensions of the images.
        - frames: List to store frames.
        
        Output:
        Initializes the class attributes.
        """
        self.vis_test = vis_test
        self.im_size = im_size
        self.frames = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Method: on_epoch_end(self, epoch, logs=None)
        
        Description:
        This method is automatically called at the end of each training epoch. It generates 
        a frame that includes both the original and the predicted images, and appends this 
        frame to the frames list.
        
        Inputs:
        - epoch: The current epoch number.
        - logs: Dictionary of logs (automatically provided by Keras, contains metrics and other information).
        
        Output:
        Appends a new frame to the frames list.
        """
        preds_test = self.model.predict(self.vis_test)
        
        fig, axes = plt.subplots(2, 8, figsize=(16,4))
        
        plt.suptitle(f'Epoch: {epoch+1}')
        
        for j in range(8):
            # Plot original image
            axes[0, j].imshow(self.vis_test[j].reshape(self.im_size, self.im_size), cmap='gray')
            axes[0, j].axis('off')

            
            # Plot predicted image
            axes[1, j].imshow(preds_test[j].reshape(self.im_size, self.im_size), cmap='gray')
            axes[1, j].axis('off')
   
                
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frame = imageio.imread(buf)
        
        self.frames.append(frame)

        
        plt.close(fig)
        buf.truncate(0)
        buf.seek(0)