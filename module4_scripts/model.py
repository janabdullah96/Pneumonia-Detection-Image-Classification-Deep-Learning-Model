from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
from types import SimpleNamespace
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

np.random.seed(123)


class Model:

    """Base Model class"""
    
    def __init__(self, model_n, labels, **kwargs):
        """
        Params:
            model_n: str
                model run number 
                (to keep track of model .h5 files)
            labels: list
                list of class labels
                Should be in order of integer label
                i.e. if Class1 is denoted by 0 and 
                Class2 is denoted by 1, the list passed
                in should be [Class1, Class2]
            kwargs: dict
                all preprocessed inputs
                i.e. all 2D and 4D shaped train, 
                test and validation sets
        """
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.model_n = model_n
        self.labels = labels
        self.model = None
        self.best_model = None
        return
    
    def build(self, layers, lr=0.01):
        """
        Method to compile a sequential network

        Params:
            layers: list
                list of layers to add to model
            lr: float
                learning rate for gradient descent
                (default = 0.01)
        """
        print(f"Initializing Sequential model...")
        self.model = Sequential()
        for idx, layer in enumerate(layers, 1):
            self.model.add(layer)
            print(f"Added model layer {idx}")
        print(
            "\nCompiling Model:",
            "\n\tOptimizer = Stochastic Gradient Descent with Momentum",
            "\n\tLoss Function = Binary Crossentropy",
            "\n\tMetrics = Accuracy, Precision, Recall, AUC\n"
        )
        self.model.compile(
            optimizer=SGD(lr=lr), 
            loss='binary_crossentropy', 
            metrics=[
                BinaryAccuracy(name='accuracy'),
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc')
            ]
        )
        print("Completed model compilation.")
    
    def train(self, **kwargs):
        """
        Method to train the network
        
        Keyword Params:
        
            epochs: int
                number of epochs (default = 150)
            batch_size: int
                batch size (default = 100)
            steps_per_epoch: int
                steps per epoch (default = 100)
            weighted: bool
                Toggle whether to apply balanced class weights
                (default = True)
            workers: int
                number of workers to activate (default = 8)
            lr_reduction_factor: float
                factor for ReduceLROnPlateu callback
            lr_patience: float
                patience for ReduceLROnPlateu callback
            early_stopping_patience:
                patience for EarlyStopping callback
        """
        params = {
            "epochs": 150,
            "batch_size": 100,
            "steps_per_epoch": 100,
            "weighted": True,
            "workers": 8,
            "lr_reduction_factor": 0.5,
            "lr_patience": 7,
            "early_stopping_patience": 14
        }
        params.update(kwargs)
        n = SimpleNamespace(**params)
        print(f'Training Params: {json.dumps(params, indent=4)}\n')

        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=n.lr_reduction_factor, 
                verbose=1, 
                patience=n.lr_patience
                ),
            EarlyStopping(
                monitor='val_loss', 
                patience=n.early_stopping_patience
                ),
            ModelCheckpoint(
                filepath=f"model/best_model_{self.model_n}.h5", 
                monitor='val_loss', 
                save_best_only=True
                )
            ]

        if n.weighted:
            class_weights = {}
            for idx, class_ in enumerate(self.labels):
                weight = (1/sum(self.y_train == idx)) * len(self.y_train) / 2
                if isinstance(weight, list) or isinstance(weight, np.ndarray):
                    class_weights[idx] = weight[0]
                else:
                    class_weights[idx] = weight
            class_weights_display = {self.labels[k]: v for k, v in class_weights.items()}
            print(f'Class Weights: {json.dumps(class_weights_display, indent=4)}\n')
        else:
            class_weights = None
        
        print("="*70)
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=n.epochs,
            batch_size=n.batch_size,
            steps_per_epoch=n.steps_per_epoch,
            callbacks=callbacks,
            validation_data=(self.X_val, self.y_val),
            class_weight=class_weights,
            workers=n.workers
        )
        self.__display_training_performance(history.history)
        self.best_model = load_model(f"model/best_model_{self.model_n}.h5")

    def evaluate(self):
        """
        Method to display evaluations for each set using the best model saved.
        Displays the loss function and metric values 
        as well as confusion matrices for each set
        """
        print("Evaluating best model...\n")
        self.__evaluate_set("Train", self.X_train, self.y_train)
        self.__evaluate_set("Validation", self.X_val, self.y_val)
        self.__evaluate_set("Test", self.X_test, self.y_test)
            
    def __display_training_performance(self, history):
        print("\n\n\t\t=== Displaying Training Performance Metrics ===")
        #validation and training metrics will be plotted together
        #so we only want the unique metric names
        metrics = [x for x in history.keys() if 'val' not in x and x != 'lr']
        fig, ax = plt.subplots(len(metrics))
        fig.set_size_inches(9, 6*len(metrics))
        for i, metric in enumerate(metrics):
            ax[i].set_title(f"{metric.title()} Curve")
            ax[i].plot(history[metric], label=f"Training {metric}", lw=1.5)
            ax[i].plot(history[f"val_{metric}"], label=f"Validation {metric}", lw=1.5)
            ax[i].set_ylim(0, 1.05)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.xlabel('Epochs', fontsize=14)
        plt.show()

    def __evaluate_set(self, set_, X, y):
        print("="*100)
        print(f"=== Evaluating best model on {set_.title()} set ===\n")
        loss, acc, precision, recall, auc = self.best_model.evaluate(X, y, verbose=0)
        y_pred = self.best_model.predict(X)
        y_pred_binary = [round(j, 0) for i in y_pred for j in i]
        print(
            "Classification Report\n",
            classification_report(
                y, y_pred_binary, labels=[0, 1], target_names=self.labels
            ),
            f"\n\nLoss: {loss}",
            f"\nAccuracy: {acc}"
        )
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(12, 6)
        
        #plot confusion matrix
        sns.heatmap(
            confusion_matrix(y, y_pred_binary), 
            cmap='Blues', 
            annot=True, 
            fmt='.6g', 
            xticklabels=self.labels, 
            yticklabels=self.labels, 
            cbar=False,
            ax=ax[0]
        )
        ax[0].set_title(f"{set_} Set Confusion Matrix\n", fontsize=14)
        plt.tight_layout()

        #plot ROC curve
        fp, tp, _ = roc_curve(y, y_pred)
        ax[1].plot(fp, tp, label=f"AUC: {auc}")
        ax[1].set_xlabel('FP')
        ax[1].set_ylabel('TP')
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1.05)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].set_title(f"{set_} Set ROC Curve", fontsize=14)
        ax[1].legend()
        plt.tight_layout()
        
        plt.show()


class DenseModel(Model):
    
    """
    Model child class constructing densely connected networks
    These networks are built using 2D-shaped data inputs
    """

    def __init__(self, model_n, labels, **kwargs):
        super().__init__(model_n, labels, **kwargs)
        self.X_train, self.y_train = self.X_train_2D, self.y_train_2D
        self.X_test, self.y_test = self.X_test_2D, self.y_test_2D
        self.X_val, self.y_val = self.X_val_2D, self.y_val_2D
        return


class ConvolutionalModel(Model):
    
    """
    Model child class for constructing networks with a convolutional base
    These networks are built using 4D-shaped data inputs
    """

    def __init__(self, model_n, labels, **kwargs):
        super().__init__(model_n, labels, **kwargs)
        self.X_train, self.y_train = self.X_train_4D, self.y_train_4D
        self.X_test, self.y_test = self.X_test_4D, self.y_test_4D
        self.X_val, self.y_val = self.X_val_4D, self.y_val_4D
        return

        