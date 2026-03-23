#!/home/marko/stfc/dials_build/conda_base/bin/python3
from dynamic.spots import SpotsList
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from scipy.stats import pearsonr
from typing import List
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt


FEATURE_NAMES = [
    'kx', 'ky', 'kz', 'z', 'intensity', 'sigma', 'resolution',
    'average_Fo', 'median_Fo', 'n_spots', 'i_max', 'i_min',
    'i_mean', 'i_90', 'iqr', 'max_res', 'min_res', 'mean_res',
    'n_neighbor', 'nn_avg_intensity', 'nn_max_intensity', 'nn_rel_intensity'
]


class LocalFitter:

    def __init__(self, datasets: List[SpotsList] = None,
                 n_estimators=600,
                 learning_rate=0.05,
                 max_depth=5,
                 random_state=1,
                 xy_distance_cutoff=100.0):

        self.datasets = datasets
        self.feature_names = FEATURE_NAMES

        total_spot_counter = 0
        if datasets:
            for idx, dataset in enumerate(self.datasets):
                for spot in dataset:
                    total_spot_counter += 1

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_train_spots = total_spot_counter
        self.distance_cutoff = xy_distance_cutoff
        self.prefix = 'local'

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
                                      xy_distance_cutoff=self.distance_cutoff)

                    kx = spot.kx[0][0]
                    ky = spot.ky[0][0]
                    kz = spot.kz[0][0]
                    x.append([kx, ky, kz, spot.z,
                              spot.intensity, spot.sigma,
                              spot.resolution, dataset.average_Fo,
                              dataset.median_Fo, en.n_spots, en.i_max,
                              en.i_min, en.i_mean, en.i_90, en.iqr,
                              en.max_res,
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

    def apply_fc_model(self, spots_list: SpotsList):

        model = self.model

        group = spots_list.group_by_image()

        for key in group.keys():
            spots_on_image = group[key]
            spots_sorted = sorted(spots_on_image,
                                  key=lambda spot: spot.resolution)

            for index, spot in enumerate(spots_sorted):

                en = LocalSpotEnv(index, spots_sorted,
                                  xy_distance_cutoff=self.distance_cutoff)

                kx = spot.kx[0][0]
                ky = spot.ky[0][0]
                kz = spot.kz[0][0]

                x = np.array([kx, ky, kz, spot.z,
                              spot.intensity, spot.sigma,
                              spot.resolution, spots_list.average_Fo,
                              spots_list.median_Fo, en.n_spots, en.i_max,
                              en.i_min, en.i_mean, en.i_90, en.iqr, en.max_res,
                              en.min_res, en.mean_res, en.n_neighbor,
                              en.nn_avg_intensity, en.nn_max_intensity,
                              en.nn_rel_intensity])
                Fo_corrected = float(model.predict([x])[0])
                spot.Fo_corrected = Fo_corrected

    def save_model(self, path=None, filename=None):

        if not filename:
            filename = f"{self.prefix}_"
            filename += f"n_{self.n_estimators:05d}_"
            filename += f"rate_{self.learning_rate:.5f}_"
            filename += f"depth_{self.max_depth:02d}_"
            filename += f"seed_{self.random_state:02d}_"
            filename += f"spots_{self.n_train_spots:06d}.joblib"
        if path:
            filename = path + '/' + filename

        payload = {
            "model": self.model,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "n_trains_spots": self.n_train_spots,
            "distance_cutoff": self.distance_cutoff,
        }

        print(f'Saving model in {filename}.')
        joblib.dump(payload, filename)

    def load_model(self, path=None, filename=None):

        if not filename:
            filename = f"{self.prefix}_"
            filename += f"n_{self.n_estimators:05d}_"
            filename += f"rate_{self.learning_rate:.5f}_"
            filename += f"depth_{self.max_depth:02d}_"
            filename += f"seed_{self.random_state:02d}_"
            filename += f"spots_{self.n_train_spots:06d}.joblib"

        if path:
            filename = path + '/' + filename

        print(f'Loading model from {filename}.')
        payload = joblib.load(filename)

        self.model = payload['model']
        self.n_estimators = payload['n_estimators']
        self.learning_rate = payload['learning_rate']
        self.max_depth = payload['max_depth']
        self.random_state = payload['random_state']
        self.n_train_spots = payload['n_trains_spots']
        self.distance_cutoff = payload['distance_cutoff']

        print(f'n_estimators: {self.n_estimators}')
        print(f'learning_rate: {self.learning_rate}')
        print(f'max depth: {self.max_depth}')
        print(f'random_state: {self.random_state}')
        print(f'n_train_spots: {self.n_train_spots}')
        print(f'distance_cutoff: {self.distance_cutoff}')

    def validate_fc_model(self):
        x, y = self.extract_training_data()

        scores = cross_val_score(self.model, x, y, cv=5,
                                 scoring='neg_mean_absolute_error')
        print("MAE scores:", -scores)
        print("Mean MAE:", -scores.mean())

        preds = self.model.predict(x)
        print("Correlation Fc_pred vs Fc_true:", pearsonr(preds, y)[0])

    # ------------------------------------------------------------------
    # Feature analysis methods
    # ------------------------------------------------------------------

    def plot_feature_importances(self, x=None, y=None):
        """Plot feature importances from the trained model.
        Trains a fresh model on (x, y) if self.model is not set."""

        if not hasattr(self, 'model'):
            if x is None or y is None:
                x, y = self.extract_training_data()
            msg = "No trained model found "
            msg += "— fitting one for importance analysis."
            print(msg)
            self.train_fc_model()

        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(
            [self.feature_names[i] for i in sorted_idx],
            importances[sorted_idx]
        )
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Feature Importances — {self.__class__.__name__}")
        plt.tight_layout()
        plt.show()

        print("\nRanked feature importances:")
        for i in reversed(sorted_idx):
            print(f"  {self.feature_names[i]:<22s}  {importances[i]:.4f}")
        return self.feature_names, importances

    def run_rfecv(self, x=None, y=None,
                  n_estimators=100, cv=5, step=1):
        """Run Recursive Feature Elimination with Cross-Validation.

        Uses a faster (fewer estimators) GBT to keep runtime reasonable.
        Stores the fitted RFECV selector as self.rfecv and prints/plots
        results.

        Parameters
        ----------
        x, y        : training arrays; extracted automatically if not provided
        n_estimators: estimator tree count for the internal fast model
        cv          : number of cross-validation folds
        step        : number of features removed per iteration
        """

        if x is None or y is None:
            x, y = self.extract_training_data()

        fast_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

        rfecv = RFECV(
            estimator=fast_model,
            step=step,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )

        print("Running RFECV — this may take a while...")
        t1 = time.time()
        rfecv.fit(x, y)
        t2 = time.time()
        print(f"RFECV done in {(t2 - t1):.2f} sec")
        print(f"Optimal number of features: {rfecv.n_features_}")

        self.rfecv = rfecv

        print("\nFeature selection results:")
        for name, selected, rank in zip(self.feature_names,
                                        rfecv.support_,
                                        rfecv.ranking_):
            marker = '✓' if selected else '✗'
            print(f"  {marker}  (rank {rank:2d})  {name}")

        # Plot CV scores vs number of features
        fig, ax = plt.subplots(figsize=(8, 4))
        nrange = len(rfecv.cv_results_['mean_test_score']) + 1
        n_features_range = range(1, nrange)
        mean_scores = -rfecv.cv_results_['mean_test_score']
        std_scores = rfecv.cv_results_['std_test_score']

        ax.plot(n_features_range, mean_scores, marker='o', label='Mean MAE')
        ax.fill_between(n_features_range,
                        mean_scores - std_scores,
                        mean_scores + std_scores,
                        alpha=0.2, label='±1 std')
        ax.axvline(rfecv.n_features_, color='red', linestyle='--',
                   label=f'Optimal: {rfecv.n_features_} features')
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("MAE")
        ax.set_title(f"RFECV — {self.__class__.__name__}")
        ax.legend()
        plt.tight_layout()
        plt.show()

        return rfecv

    def run_tpot(self, x=None, y=None,
                 use_rfecv_mask=True,
                 generations=5,
                 population_size=20,
                 export_path='best_pipeline.py'):
        """Run TPOT AutoML on the (optionally RFECV-pruned) feature set.

        Parameters
        ----------
        x, y : training arrays; extracted automatically if not provided
        use_rfecv_mask : if True and self.rfecv exists, prune features first
        generations : TPOT generations to evolve
        population_size : TPOT population size
        export_path : path to export the winning sklearn pipeline
        """

        try:
            from tpot import TPOTRegressor
        except ImportError:
            raise ImportError("TPOT is not installed. Run: pip install tpot")

        if x is None or y is None:
            x, y = self.extract_training_data()

        if use_rfecv_mask and hasattr(self, 'rfecv'):
            x = x[:, self.rfecv.support_]
            selected = [n for n, s in zip(self.feature_names,
                                          self.rfecv.support_) if s]
            msg = "Using RFECV-pruned feature set "
            msg + f"({len(selected)} features): {selected}"
            print(msg)
        else:
            if use_rfecv_mask:
                print("use_rfecv_mask=True but no RFECV result found — "
                      "run run_rfecv() first, or pass use_rfecv_mask=False.")
            msg = "Using full feature set "
            msg += f"({len(self.feature_names)} features)"
            print(msg)

        tpot = TPOTRegressor(
            generations=generations,
            population_size=population_size,
            scoring='neg_mean_absolute_error',
            random_state=self.random_state,
            verbosity=2,
            n_jobs=-1
        )

        print("Running TPOT — this will take a while...")
        t1 = time.time()
        tpot.fit(x, y)
        t2 = time.time()
        print(f"TPOT done in {(t2 - t1):.2f} sec")

        if export_path:
            tpot.export(export_path)
            print(f"Best pipeline exported to {export_path}")

        self.tpot = tpot
        return tpot

# ------------------------------------------------------------------
# Helper classes / functions (unchanged)
# ------------------------------------------------------------------


class LocalSpotEnv:

    def __init__(self, spot_index, spots_list, xy_distance_cutoff=20.0):

        self.n_spots = len(spots_list)
        selected_spot = spots_list[spot_index]

        intensities = np.sort([spot.intensity for spot in spots_list])
        resolutions = np.sort([spot.resolution for spot in spots_list])

        neighbors = find_neighbors_in_xy(spot_index, spots_list,
                                         xy_distance_cutoff)

        self.i_max = intensities.max()
        self.i_min = intensities.min()
        self.i_mean = intensities.mean()
        self.i_90 = np.percentile(intensities, 90)
        self.iqr = (np.percentile(intensities, 75) -
                    np.percentile(intensities, 25))
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
        else:
            self.nn_avg_intensity = 0.0
            self.nn_max_intensity = 0.0
            self.nn_rel_intensity = 0.0


def compute_xy_px_distance(spot1, spot2):
    return np.sqrt((spot1.x - spot2.x)**2 + (spot1.y - spot2.y)**2)


def find_neighbors_in_xy(spot_index, spots_list, xy_distance_cutoff):

    selected_spot = spots_list[spot_index]
    distances = np.array([compute_xy_px_distance(selected_spot, spot)
                          for spot in spots_list])

    neighbors = []
    for idx, (spot, distance) in enumerate(zip(spots_list, distances)):
        if distance < xy_distance_cutoff and idx != spot_index:
            neighbors.append(spot)

    return neighbors
