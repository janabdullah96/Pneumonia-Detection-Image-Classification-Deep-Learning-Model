
import os
import shutil
import random


class DataDirectoryConstructor:
    
    """
    class to construct splits of train, test, and validation sets
    in data directory 
    """
    
    SPLITS_DIRS = ["train", "test", "val"]   
    
    def __init__(self, directory, subdirs, train_size, test_size, val_size):
        """
        Params
            
            directory: str
                origin directory
            subdirs: list
                list of subdirectories (i.e. classes)
            train_size: float
                train set split size
            test_size: float
                test set split size
            val_size: float
                validation set split size
        """
        self.origin_dir = directory
        self.subdirs = subdirs
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        return
    
    def split_dataset(self):
        self.display_params()
        self.make_dirs()
        all_files = self.__label_files()
        train, test, val = self.__generate_sets(all_files)
        self.__create_set_directory("train", train)
        self.__create_set_directory("test", test)
        self.__create_set_directory("val", val)
        
    def display_params(self):
        print("Class Directories:")
        for class_ in self.subdirs:
            print(f"\t{self.origin_dir}{class_}/")
        print(
            "\nSplits:",
            f"\n\tTrain size: {self.train_size}",
            f"\n\tTest size: {self.test_size}",
            f"\n\tValidation size: {self.val_size}\n"
        )

    def make_dirs(self):
        """
        Method to create train, test, and val subdirectories.
        If subdirectories already exist, they will be cleared
        """
        for split_dir in self.SPLITS_DIRS:
            dir_ = self.origin_dir + split_dir
            if os.path.exists(dir_):
                shutil.rmtree(dir_)
            os.mkdir(dir_)
            for subdir in self.subdirs:
                os.mkdir(dir_ + "/" + subdir)

    def __label_files(self):
        """
        Method to generate a complete list of files and their
        corresponding class labels. (i.e. a tuple is created for 
        each file in the format (label, file))
        """
        all_files = []
        for subdir in self.subdirs:
            #Subdirs are also label names
            all_files.extend(
                [(subdir, file) for file in os.listdir(self.origin_dir+subdir)]
            )
        random.shuffle(all_files)
        return all_files
    
    def __generate_sets(self, file_ls):
        """
        Method to allocate each file in the directory to the 
        training, test, and validation sets subdirectories
        """
        n = len(file_ls)
        train = []
        test = []
        val = []
        n_train = round(n*self.train_size, 0)
        n_test = round(n*self.test_size, 0)
        n_val = round(n*self.val_size, 0)
        while n_train != 0:
            train.append(file_ls[0])
            file_ls = file_ls[1:]
            n_train-=1
        while n_test != 0:
            test.append(file_ls[0])
            file_ls = file_ls[1:]
            n_test-=1
        while n_val > 1:
            val.append(file_ls[0])
            file_ls = file_ls[1:]
            n_val-=1
        return train, test, val
        
    def __create_set_directory(self, set_, set_files_ls):
        """
        Method to create the final directories
        Class subdirectories will be created for each 
        set and relevant files will be copied there
        """
        set_dir = self.origin_dir + set_
        for class_, file in set_files_ls:
            src_dir = self.origin_dir + class_ + "/" + file
            dest_dir = self.origin_dir + set_ + "/" + class_ + "/" + file
            shutil.copy(src_dir, dest_dir)
        print(f"Generated {set_.title()} data directory")
   
   