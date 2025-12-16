#!/home/marko/stfc/dials_build/conda_base/bin/python3
from dynamic.spots import SpotsList
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import numpy as np


class Fitter:

    def __init__(self, spots: SpotsList):

        self.spots = spots.spots
        self.scale = spots.scale

    def extract_training_data(self):

        x = []
        y = []

        for spot in self.spots:

            x.append([spot.H, spot.K, spot.L, spot.z,
                      spot.intensity, spot.sigma, self.scale,
                      spot.resolution,
                      ])

            y.append(spot.Fc)

        return np.array(x), np.array(y)

    def train_fc_model(self):

        x, y = self.extract_training_data()

        model = GradientBoostingRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            random_state=0
        )

        model.fit(x, y)
        self.model = model

    def validate_fc_model(self):
        x, y = self.extract_training_data()

        # MAE (mean absolute error)
        scores = cross_val_score(self.model, x, y, cv=5,
                                 scoring='neg_mean_absolute_error')
        print("MAE scores:", -scores)
        print("Mean MAE:", -scores.mean())

        preds = self.model.predict(x)
        print("Correlation Fc_pred vs Fc_true:", pearsonr(preds, y)[0])

    def predict_fc_for_spot(self, spot, global_scale):
        model = self.model
        x = np.array([[spot.H, spot.K, spot.L, spot.z, spot.intensity,
                       spot.sigma, global_scale, spot.resolution
                       ]])
        return float(model.predict(x)[0])

    def apply_fc_model(self, spots_list):
        for spot in spots_list.spots:
            predicted = self.predict_fc_for_spot(spot, spots_list.scale)
            spot.Fo_corrected = predicted
