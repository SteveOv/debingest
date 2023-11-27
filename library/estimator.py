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

    # These need to match up to the under Scikit-Learn model
    __features_scale = [1., 1., 1., 0.01, 1., 1., 1., 1.]
    __features = ["rA_plus_rB", "k", "bA", "inc", "ecosw", "esinw", "J", "L3"]
    __sigmas = [f"{f}_sigma" for f in __features]

    def __init__(self):
        # Suppress annoying TF info messages
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        this_dir = Path(getsourcefile(lambda:0)).parent

        model_file = f"{this_dir}/cnn_model.h5"
        print(f"Estimator is loading model file {model_file}")
        self.model = load_model(model_file)

    @property
    def features(self) -> list:
        """
        The list of predicted features, in the order returned by the model.
        """
        return self.__features

    def predict(self, fold_mags: np.ndarray, iterations: int=1000) -> DataFrame:
        """
        Predicts the estimated values of the passed folds.

        :fold_mags: 1+ folded reduced light-curves in shape[#LCs, 1024, 1]
        :iterations: number of MC predictions or < 1 for a simple prediction.
        :returns: pandas DataFrame with predicted features in columns
        """
        iterations = max(1, iterations)
        dropout = iterations > 1
        print(f"Making predictions for {fold_mags.shape[0]} light-curves",
              f"over {iterations} MC Dropout iterations" if dropout else "")

        # If dropout, we make multiple predictions for each LC with training
        # switched on so that each prediction is with a statistically unique
        # "dropout" subset of the model's net: this is the MC Dropout algorithm.
        # The predictions are given in shape[#iterations, #LCs, #features]
        mc_preds = np.stack([
            self.model(fold_mags, training=dropout)
            for ix in range(iterations)
        ])

        # Summary stats over the iterations axis & undo feature scaling.
        # For each LC we want predictions followed by 1-sigmas on the same axis
        # so we get the final predictions in shape [#LCs, #feature + #sigmas]
        unscaled_preds = np.divide(mc_preds, self.__features_scale)
        preds = np.concatenate([
            np.mean(unscaled_preds, axis=0),
            np.std(unscaled_preds, axis=0)
        ],
        axis=1)

        col_names = np.concatenate([
            self.__features,
            self.__sigmas
        ])

        # Load into a Pandas DataFrame
        return DataFrame(data=preds, columns=col_names)
