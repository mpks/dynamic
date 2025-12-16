#!/home/marko/stfc/dials_build/conda_base/bin/python3
from dynamic.kinematic import get_resolution_calculator
from dynamic.kinematic import StructureFactorCalculator
from multiprocessing import Pool
import numpy as np
from dynamic.io import read_dials_hkl
from typing import List
from dials.array_family import flex
import matplotlib.pyplot as plt


class Spot:

    def __init__(self, H, K, L, intensity, sigma, z=None,
                 resolution=None, Fc=None, Fo_corrected=None):

        self.H = H
        self.K = K
        self.L = L
        self.z = z
        self.miller = (H, K, L)
        self.intensity = intensity
        self.sigma = sigma
        self.resolution = resolution
        self.Fc = Fc
        self.Fo = None
        self.Fo_scaled = None
        self.scale_correct = None
        self.Fo_corrected = Fo_corrected


class SpotsList:

    def __init__(self,
                 spots: List[Spot],
                 material: str = 'paracetamol',
                 output_prefix: str = '0000',
                 ) -> None:

        if material == 'paracetamol':
            cif_file = '/home/marko/active/dyn/data/our_paracetamol.cif'
        else:
            raise ValueError('Unknown material: ', material)
        self.material = material
        self.cif_file = cif_file
        self.spots = spots
        self.output_prefix = output_prefix
        self.scale = 1.0

    @classmethod
    def from_refl(cls,
                  refl_file: str,
                  material: str = 'paracetamol',
                  output_prefix: str = '0000',
                  fitted_profile: bool = True):

        refl = flex.reflection_table.from_file(refl_file)

        spots = []

        hkl_flex = refl["miller_index"]
        hkl_list = [hkl_flex[i] for i in range(len(hkl_flex))]

        if fitted_profile:
            intensities = list(refl["intensity.prf.value"])
            sigmas = list(refl["intensity.prf.variance"])
        else:
            intensities = list(refl["intensity.sum.value"])
            sigmas = list(refl["intensity.sum.variance"])

        vals = list(refl["xyzobs.px.value"])       # list of floats

        for i in range(len(hkl_list)):

            H, K, L = hkl_list[i]
            x, y, z = vals[i]
            intensity = intensities[i]
            sigma = sigmas[i]
            spot = Spot(H, K, L, intensity, sigma=sigma, z=int(z))
            spots.append(spot)

        return cls(spots, material=material, output_prefix=output_prefix)

    @classmethod
    def from_hkl(cls,
                 hkl_filename: str,
                 material: str = 'paracetamol',
                 output_prefix: str = '0000'
                 ):

        Idict, sigma_dict = read_dials_hkl(hkl_filename)

        spots = []
        for miller in Idict:
            h, k, ll = miller
            intensity = Idict[miller]
            sigma = sigma_dict[miller]
            spot = Spot(h, k, ll, intensity, sigma)
            spots.append(spot)

        return cls(spots, material=material, output_prefix=output_prefix)

    def save_to_hkl(self, filename='output.hkl'):

        with open(filename, 'w') as f:

            for spot in self.spots:
                H, K, L = spot.miller
                if spot.intensity > 0:
                    intensity = (spot.Fo_corrected * self.scale)**2
                # else:
                #     intensity = spot.intensity
                    sig = spot.sigma
                    line = f"{H:4d}{K:4d}{L:4d}  {intensity:12.4f}  "
                    line += f"{sig:10.4f}\n"
                    f.write(line)

    def compute_Fo_and_bulk_scale(self):

        Fo = []
        Fc = []
        for spot in self.spots:
            if spot.intensity > 0:
                fo = np.sqrt(spot.intensity)
                spot.Fo = fo
                Fo.append(fo)
                Fc.append(spot.Fc)

        Fo = np.array(Fo)
        Fc = np.array(Fc)

        scale = Fo.mean() / Fc.mean()
        self.scale = scale
        print(f' Computed global scale: {scale:.2f}')
        print(' Scaling spots')
        for spot in self.spots:
            if spot.intensity > 0:
                spot.Fo_scaled = spot.Fo / scale
            else:
                spot.Fo_scaled = None

    def compute_fc(self):

        params = []
        for spot in self.spots:
            params.append((self.cif_file, spot))

        print(' Computing the Fc')
        with Pool(processes=20) as pool:
            fcs = pool.map(sf_spot_multiprocess, params)

        for i in range(len(fcs)):
            self.spots[i].Fc = fcs[i]

    def compute_resolutions(self):

        resolution_func = get_resolution_calculator(self.cif_file)

        for spot in self.spots:
            spot.resolution = resolution_func(spot.miller)

    def plot_fc_vs_predicted(self, filename=None):

        true_fc = []
        pred_fc = []

        for spot in self.spots:
            true_fc.append(spot.Fc)
            pred_fc.append(spot.Fo_corrected)

        fig = plt.figure(figsize=(3.375, 3.0))
        ax = fig.add_axes([0.13, 0.14, 0.84, 0.82])
        plt.plot([min(true_fc), max(true_fc)],
                 [min(true_fc), max(true_fc)],
                 color='#BEBEBE', lw=0.5)

        ax.plot(true_fc, pred_fc, marker='o', ms=1.5, c='C0', mew=0, lw=0)

        ax.set_xlabel("True Fc")
        ax.set_ylabel("Predicted Fc")

        if not filename:
            filename = f"{self.output_prefix}_Fc_vs_Fpred.png"
        plt.savefig(filename, dpi=400)


def sf_spot_multiprocess(params):
    cif_file, spot = params
    sf = StructureFactorCalculator(cif_file)
    Fc = abs(sf.structure_factor(spot.miller))
    return Fc


def sf_multiprocess(params):
    cif_file, miller = params
    sf = StructureFactorCalculator(cif_file)
    return abs(sf.structure_factor(miller))


def double_exp(x, a, b, c, d):
    return a*np.exp(b*x) + c*np.exp(d*x)
