# Pneumonia-Detection-Image-Classification-Deep-Learning-Model
Objective: To build a deep learning neural network model that predicts/classifies whether a patient has pneumonia using chest X-ray images as model inputs.

## Repo Contents
### Python Modules (in 'module4_scripts' folder)

A python package was created to serve as the infrastructure for this project.

1. dir_constructor.py - contains functionality for constructing a data directory where the images in the original dataset are split into train, test, and validation sets.
2. reader.py - contains functionality for reading image data.
3. preprocessor.py - contains functionality for converting images into data formats (2D and 4D normalized numerical arrays) that are suitable for machile learning modeling.
4. model.py - contains functionality to compile, train, and evaluate densely-connected and convolutional neural network models.
Jupyter Notebooks

### Jupyter Notebooks
1. preprocessing.ipynb - this notebook contains all the data preprocessing work, which entails reading image data and converting to 2D and 4D numpy arrays for neural network modeling compatibility. 
2. modeling.ipynb - this notebook contains all the modeling iterations. 

### *Not Included in this GitHub Repo

The below folders are in the local drive for this project but not added to the GitHub repo due to file size limitations.

1. data - this folder contains all the raw data as well as the constructed sets for this project. Raw data is obtained from the Kaggle link here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
2. model - this folder contains all the .h5 model files saved via the ModelCheckpoint callback in Keras.
3. npy - this folder contains a .npz file which is a externally saved compressed file of numpy arrays. The numpy arrays are the train, test, and validation sets for images and labels.

## The Final Model

The best model found in this project after much iteration and variation is a convolutional neural network with 1 convolving layer and 2 dense layers. See the model architecture below. 

![alt text](https://github.com/janabdullah96/Pneumonia-Detection-Image-Classification-Deep-Learning-Model/blob/main/images/model_summary.PNG)

See below the loss and accuracy curves over epochs during model training. Convergence occured around 40-45 epochs. 

![alt text](https://github.com/janabdullah96/Pneumonia-Detection-Image-Classification-Deep-Learning-Model/blob/main/images/model_training_curves.PNG)

See below model evaluation on the test set.

![alt text](https://github.com/janabdullah96/Pneumonia-Detection-Image-Classification-Deep-Learning-Model/blob/main/images/model_evaluation.PNG)

On the test set, the model achieved a 96.08% accuracy score with a 97% recall score. The false negative rate for pneumonia diagnosis is 2.57%. 

To summarize some points made in the modeling.ipynb notebook, for this domain we should focus more on the false negative rate metric amongst other evaluation metrics. Reason being we are building a model in a medical domain, and usually it's better and safer for a patient to be treated for an issue they don't have, than not being treated at all for an issue they do have. Of course, this varies on a case-by-case basis, depending on other factors such as risk of treatment. In the case of pneumonia, a common treatment is antibiotics. Taking antibiotics can be deemed a low risk treatment, so a false positive can be okay. Eventually, after checkups, it's likely that the doctor will stop prescribing the antibiotics anyways. This case is much better than if a patient has pneumonia but doesn't get treated at all, which can cause life-threatening side effects.  

### Future Work

The biggest bottleneck for this project, and generally for any deep learning project, was resource availabilty. Building deep learning models is a resource-intense process, and the availabilty of such resources determines the scalability of the project. For this project, I've temporarily subscribed to a premium Google Colab service, giving me priority access to Google's GPUs. Priority access, however, does not mean unrestricted. If institutional-level unrestricted access to resources were available, this project could have been scaled much further to try out even more iterations of networks with endless variations of parameters, nodes, layers, regularization parameters, pretrained networks etc. The infrastructure (i.e python package) created for this project should be suitable for higher-scale projects as well. 
