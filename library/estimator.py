"""
A fitting params estimator specific to the ML model used for predictions.
"""
import os
from pathlib import Path
from inspect import getsourcefile
import numpy as np
from pandas import DataFrame
from tensorflow.keras.models import load_model

class Estimator():
    """
    A fitting params estimator specific to the ML model used for predictions.
    Publishes the estimations in a pandas DataFrame, with columns for each
    feature so that client code does not need to know the specifics.

    The plan is for the training project to provide the implementation of
    this class to abstract the nuts and bolts of interacting with its model.
    """
    def __init__(self):
        # Suppress annoying TF info messages
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        this_dir = Path(getsourcefile(lambda:0)).parent

        model_file = f"{this_dir}/cnn_model.h5"
        print(f"Estimator is loading model file {model_file}")
        self.model = load_model(model_file)

    @property
    def __features_scale(self) -> list:
        return [1., 1., 1., 0.01, 1., 1., 1., 1.]

    @property
    def features(self) -> list:
        """
        The list of predicted features, in the order returned by the model.
        """
        return ["rA_plus_rB", "k", "bA", "inc", "ecosw", "esinw", "J", "L3"]

    def predict(self, fold_mags: np.ndarray) -> DataFrame:
        """
        Predicts the estimated values of the passed folds.

        :fold_mags: 1+ folded reduced light-curves in shape[#LCs, 1024, 1]
        :returns: pandas DataFrame with predicted features in columns
        """
        # The predictions are give in shape[#LCs, #features]
        print(f"Making predictions for {fold_mags.shape[0]} light-curves")

        # Undo any scaling of the features
        preds = np.divide(self.model.predict(fold_mags), self.__features_scale)

        # Load into a Pandas DataFrame
        return DataFrame(data=preds, columns=self.features)
