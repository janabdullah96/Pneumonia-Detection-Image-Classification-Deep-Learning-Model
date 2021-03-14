
import numpy as np
from os import listdir


class Preprocessor:

    """
    Class to preprocess image data into a format suitable for modeling
    """
    
    def __init__(self, **data_generators):
        """
        Params:

            data_generators: dict
                keys = name of generator
                values = ImageDataGenerator instance object
        """
        for k in data_generators:
            setattr(self, k, data_generators[k])
        self.X_train_4D, self.y_train_4D = None, None
        self.X_test_4D, self.y_test_4D = None, None
        self.X_val_4D, self.y_val_4D = None, None
        self.X_train_2D, self.y_train_2D = None, None
        self.X_test_2D, self.y_test_2D = None, None
        self.X_val_2D, self.y_val_2D = None, None
        return
    
    def preprocess(self, augment_data=False):
        """
        method to preprocess all image and label sets

        Params:
            
            augment_data: bool
                Toggle whether to append augmented training data to
                training set (default=False)
                Enabling this would double the size of the training set 
        """
        self.X_train_4D, self.y_train_4D = next(self.train_generator)
        self.X_test_4D, self.y_test_4D = next(self.test_generator)
        self.X_val_4D, self.y_val_4D = next(self.val_generator)
        if augment_data:
            print('Appending augmented data to Train set')
            X_train_aug, y_train_aug = next(self.train_generator_augmented)
            #Add the augmented data to the existing training sets.
            #This will double the training set.
            self.X_train_4D = np.concatenate((self.X_train_4D, X_train_aug))
            self.y_train_4D = np.concatenate((self.y_train_4D, y_train_aug))
            
        self.X_train_2D, self.y_train_2D = self.__reshape_set('Train', self.X_train_4D, self.y_train_4D)
        self.X_test_2D, self.y_test_2D = self.__reshape_set('Test', self.X_test_4D, self.y_test_4D)
        self.X_val_2D, self.y_val_2D = self.__reshape_set('Validation', self.X_val_4D, self.y_val_4D)

    def __reshape_set(self, set_, X, y):
        """
        Method to reshape arrays into 2D tensors
        """
        def __display_shapes(version):
            print(
                f"{version} shapes:",
                f"\n\tImages: {X.shape}",
                f"\n\tLabels: {y.shape}"
            )
            
        print(f"\n=== Processing {set_} set ===")
        __display_shapes("Original")
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        __display_shapes("Reshaped")
        return X, y

    def save_arrays(self, path):
        print(f'Compressing and saving all sets to {path}.npz')
        np.savez_compressed(
            path,
            **{
                k: v 
                for k, v in vars(self).items()
                if 'generator' not in k
                }
        )
        print('Done')
    
    @staticmethod
    def load_arrays(path):
        print(f"Reading file in path: {path}.npz")
        loaded_npz = np.load(path)
        loaded_arrays = {}
        for arr in loaded_npz.files:
            print(f"Loading set {arr}")
            loaded_arrays[arr] = loaded_npz[arr]
        return loaded_arrays

