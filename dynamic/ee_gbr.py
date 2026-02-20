#!/home/marko/stfc/dials_build/conda_base/bin/python3
from dynamic.spots import SpotsList
from sklearn.ensemble import GradientBoostingRegressor
from dynamic.gradient_boosting_regressor import GlobalFitter
from typing import List
import numpy as np
import time

"""Gradient Boosting Regressor, but with excitation error included"""


class LocalFitter(GlobalFitter):

    def __init__(self, datasets: List[SpotsList],
                 n_estimators=600,
                 learning_rate=0.05,
                 max_depth=5,
                 random_state=1,
                 distance_cutoff=100.0):

        self.datasets = datasets

        total_spot_counter = 0
        for idx, dataset in enumerate(self.datasets):
            for spot in dataset.spots:
                total_spot_counter += 1

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_train_spots = total_spot_counter
        self.distance_cutoff = distance_cutoff
        self.prefix = 'ee'

    def extract_training_data(self):

        x = []
        y = []

        total_spot_counter = 0
        for idx, dataset in enumerate(self.datasets):
            print(f"Extracting data for dataset {idx} / {len(self.datasets)}")

            group = dataset.group_by_image()

            for key in group.keys():
                spots_on_image = group[key]
                spots_sorted = sorted(spots_on_image,
                                      key=lambda spot: spot.resolution)

                for index, spot in enumerate(spots_sorted):
                    total_spot_counter += 1

                    en = LocalSpotEnv(index, spots_sorted,
                                      distance_cutoff=self.distance_cutoff)

                    x.append([spot.H, spot.K, spot.L, spot.z,
                              spot.intensity, spot.sigma,
                              spot.resolution, spot.excitation_error,
                              dataset.average_Fo,
                              dataset.median_Fo, en.n_spots, en.i_max,
                              en.i_min, en.i_mean, en.i_90, en.iqr, en.max_res,
                              en.min_res, en.mean_res, en.n_neighbor,
                              en.nn_avg_intensity, en.nn_max_intensity,
                              en.nn_rel_intensity])

                    y.append(spot.Fc)

        print(f"Extracted {total_spot_counter} spots for training")
        return np.array(x), np.array(y)

    def train_fc_model(self, n_estimators=600, learning_rate=0.05,
                       max_depth=5, random_state=0):

        x, y = self.extract_training_data()

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

        t1 = time.time()
        model.fit(x, y)
        t2 = time.time()
        print(f"Training done in {(t2 - t1):.2f} sec")
        self.model = model

    def predict_fc_for_spot(self, spot, average_Fo, median_Fo):
        model = self.model
        x = np.array([[spot.H, spot.K, spot.L, spot.z, spot.intensity,
                       spot.sigma, spot.resolution, average_Fo, median_Fo
                       ]])
        return float(model.predict(x)[0])

    def apply_fc_model(self, spots_list: SpotsList):

        model = self.model

        group = spots_list.group_by_image()

        for key in group.keys():
            spots_on_image = group[key]
            spots_sorted = sorted(spots_on_image,
                                  key=lambda spot: spot.resolution)

            for index, spot in enumerate(spots_sorted):

                en = LocalSpotEnv(index, spots_sorted,
                                  distance_cutoff=self.distance_cutoff)

                x = np.array([spot.H, spot.K, spot.L, spot.z,
                              spot.intensity, spot.sigma,
                              spot.resolution, spots_list.average_Fo,
                              spots_list.median_Fo, en.n_spots, en.i_max,
                              en.i_min, en.i_mean, en.i_90, en.iqr, en.max_res,
                              en.min_res, en.mean_res, en.n_neighbor,
                              en.nn_avg_intensity, en.nn_max_intensity,
                              en.nn_rel_intensity])
                Fo_corrected = float(model.predict([x])[0])
                spot.Fo_corrected = Fo_corrected


class LocalSpotEnv:

    def __init__(self, spot_index, spots_list, distance_cutoff=20.0):

        self.n_spots = len(spots_list)
        selected_spot = spots_list[spot_index]

        intensities = np.sort([spot.intensity for spot in spots_list])
        resolutions = np.sort([spot.resolution for spot in spots_list])

        neighbors = find_neighbors_in_xy(spot_index, spots_list,
                                         distance_cutoff)

        self.i_max = intensities.max()
        self.i_min = intensities.min()
        self.i_mean = intensities.mean()
        self.i_90 = np.percentile(intensities, 90)
        self.iqr = np.percentile(intensities, 75) - np.percentile(intensities,
                                                                  25)
        self.max_res = resolutions.max()
        self.min_res = resolutions.min()
        self.mean_res = resolutions.mean()

        self.n_neighbor = len(neighbors)
        if len(neighbors) > 0:
            neighbor_intensities = np.array([n.intensity for n in neighbors])
            self.nn_avg_intensity = neighbor_intensities.mean()
            self.nn_max_intensity = neighbor_intensities.max()
            self.nn_rel_intensity = (selected_spot.intensity /
                                     self.nn_max_intensity)
            self.nn_rel_intensity = 0.0
            # if np.isnan(self.nn_avg_intensity):
        else:
            self.nn_avg_intensity = 0.0
            self.nn_max_intensity = 0.0
            self.nn_rel_intensity = 0.0


def compute_xy_px_distance(spot1, spot2):
    return np.sqrt((spot1.x - spot2.x)**2 + (spot1.y - spot2.y)**2)


def find_neighbors_in_xy(spot_index, spots_list, distance_cutoff):

    selected_spot = spots_list[spot_index]
    distances = np.array([compute_xy_px_distance(selected_spot, spot) for
                          spot in spots_list])

    neighbors = []
    for idx, (spot, distance) in enumerate(zip(spots_list, distances)):
        if distance < distance_cutoff and idx != spot_index:
            neighbors.append(spot)

    return neighbors
