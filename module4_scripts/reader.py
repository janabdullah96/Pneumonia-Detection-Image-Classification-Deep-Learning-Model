
from keras.preprocessing.image import ImageDataGenerator
import os


class ImageReader:
    
    """
    Class to read images from data directories

    As a prerequiste, the appropriate data directory structure should
    have already been created. An illustration of the structure is below:
    
    data/
        train/
            {CLASS1}/
            {CLASS2}/
        test/
            {CLASS1}/
            {CLASS2}/
        val/
            {CLASS1}/
            {CLASS2}/
    """
    
    TRAIN_DIR = 'data/train'
    TEST_DIR = 'data/test'
    VAL_DIR = 'data/val'
    
    def __init__(self):
        self.train_generator = None
        self.train_generator_augmented = None
        self.test_generator = None
        self.val_generator = None
        return
    
    def read(self):
        """
        Method to create image data generators for each set 
        (including an augmented training data generator)
        """
        #create a ImageDataGenerator to generate slightly modified training data
        train_datagen_augmented = ImageDataGenerator(rotation_range=40,
                                                     width_shift_range=0.1,
                                                     height_shift_range=0.1,
                                                     shear_range=0.1,
                                                     zoom_range=0.1,
                                                     horizontal_flip=True,
                                                     fill_mode='nearest',
                                                     rescale=1./255)
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_val_datagen = ImageDataGenerator(rescale=1./255)
        
        constant_kwargs = {
            'target_size': (256, 256),
            'class_mode': 'binary'
        }
        self.train_generator = train_datagen.flow_from_directory(
            self.TRAIN_DIR, 
            batch_size=self.__get_batch_size(self.TRAIN_DIR),
            **constant_kwargs
        )
        self.train_generator_augmented = train_datagen_augmented.flow_from_directory(
            self.TRAIN_DIR, 
            batch_size=self.__get_batch_size(self.TRAIN_DIR),
            **constant_kwargs
        )
        self.test_generator = test_val_datagen.flow_from_directory(
            self.TEST_DIR,
            batch_size=self.__get_batch_size(self.TEST_DIR),
            **constant_kwargs
        )
        self.val_generator = test_val_datagen.flow_from_directory(
            self.VAL_DIR, 
            batch_size=self.__get_batch_size(self.VAL_DIR),
            **constant_kwargs
        )
        
    def display_read_summary(self):
        """
        Method to display summary information regarding the images read
        """
        #display classes
        print("\n=== Classes ===")
        for k, v in self.train_generator.class_indices.items():
            print(f"{k}: {v}")
        
        #display breakdown of sets 
        print("\n=== Directory Breakdown ===\n", "_"*40)
        sets_dir = {
            'Train': self.TRAIN_DIR,
            'Test': self.TEST_DIR,
            'Validation': self.VAL_DIR
        }
        for set_, dir_ in sets_dir.items():
            subsets_count = {
                k: len(os.listdir(dir_ + "/" + k))
                for k in os.listdir(dir_) 
                if k in self.train_generator.class_indices.keys()
            }
            n_items_set = sum(subsets_count.values())
            print(set_)
            for subset, count in subsets_count.items():
                print(f"\t{subset}:")
                print(f"\t\tCount: {count}")
                print(f"\t\tProportion: {round(count/n_items_set, 2)}")
            print("_"*40)
        
    def __get_batch_size(self, dir_):
        """
        Get size of entire set directory
        """
        subdirs = os.listdir(dir_)
        n=0
        for subdir in subdirs:
            subdir = dir_ + "/" + subdir
            if os.path.isdir(subdir):
                n+=len(os.listdir(subdir))
        return n
        
