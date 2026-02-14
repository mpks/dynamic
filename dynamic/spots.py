#!/home/marko/stfc/dials_build/conda_base/bin/python3
from dynamic.kinematic import get_resolution_calculator
from dynamic.kinematic import StructureFactorCalculator
from multiprocessing import Pool
import numpy as np
from dynamic.io import read_dials_hkl
from typing import List
from dials.array_family import flex
import matplotlib.pyplot as plt
import gemmi
from dynamic.excitation_error import ExcitationErrorCalculator
from dynamic.calc import find_best_scale
from dynamic.calc import s1_distance
from dxtbx.model.experiment_list import ExperimentListFactory
from dynamic.calc import compute_s1


class Spot:

    def __init__(self,
                 H: int,
                 K: int,
                 L: int,
                 intensity: float,
                 sigma: float,
                 x: float = None,
                 y: float = None,
                 z: int = None,
                 resolution: float = None,
                 Fc: float = None,
                 Fo_corrected: float = None,
                 excitation_error: float = None,
                 s1_x: float = None,
                 s1_y: float = None,
                 s1_z: float = None
                 ) -> None:

        """
        Spots object

        Parameters
        ----------

        H : integer
            Miller index H.
        K : integer
            Miller index K.
        L : integer
            Miller index L.

        """

        self.H = H
        self.K = K
        self.L = L
        self.x = x
        self.y = y
        self.z = z
        self.s1_x = s1_z
        self.s1_y = s1_y
        self.s1_z = s1_z
        self.miller = (H, K, L)
        self.intensity = intensity
        self.sigma = sigma
        self.resolution = resolution
        self.Fc = Fc
        self.Fo = None
        self.excitation_error = excitation_error
        self.Fo_scaled = None               # Fo scaled to the Fc scale
        self.Fo_corrected = Fo_corrected    # On the same scale as Fo

    def __str__(self):

        spot_str = f"({self.H:+3d}, {self.K:+3d}, {self.L:+3d})"
        spot_str += f" I = {self.intensity:>8.2f}, z = {self.z:04d}"
        return spot_str

    def is_miller(self, *args):

        if len(args) == 1:
            try:
                H, K, L = args[0]
            except (TypeError, ValueError):
                raise TypeError('Expected a tuple (H, K, L)')
        elif len(args) == 3:
            H, K, L = args
        else:
            raise TypeError("Expected either three params H, K, L or 3-tuple")

        return (H == self.H) and (K == self.K) and (L == self.L)

    def s1_distance(self, spot):

        return s1_distance(self, spot)


class SpotsList:

    def __init__(self,
                 spots: List[Spot],
                 material: str = 'paracetamol',
                 output_prefix: str = '0000',
                 ) -> None:

        if material == 'paracetamol':
            cif_file = '/home/marko/active/dd/data/our_paracetamol.cif'
        elif material == 'ireloh':
            cif_file = '/home/marko/active/dd/data/1870980_ireloh.cif'
        else:
            raise ValueError('Unknown material: ', material)
        self.material = material
        self.cif_file = cif_file
        self.spots = spots
        self.output_prefix = output_prefix
        self.global_scale = 1.0
        self.average_Fo = None
        self.median_Fo = None

    def __len__(self):
        return len(self.spots)

    def __iter__(self):
        return iter(self.spots)

    def __getitem__(self, index):
        return self.spots[index]

    def __setitem__(self, index, value):
        self.spots[index] = value

    def compute_excitation_errors(self, expt_file, refl_file,
                                  out_path, set_idx, exp_id=0,
                                  plot=True):
        """
        Attach excitation error to each Spot using DIALS experiment geometry.
        """

        print("Computing excitation errors...")

        ee_calc = ExcitationErrorCalculator(
            experiments_json=expt_file,
            exp_id=exp_id, refl_file=refl_file)

        ee_calc.ee()
        ee_calc.attach_to_spots_list(self)
        if plot:
            ee_calc.plot_ee_per_image(self, out_path, set_idx)

        print("Excitation error computation finished.")

    def get_by_miller(self, miller_index):

        spots = []
        for spot in self:
            if spot.is_miller(miller_index):
                spots.append(spot)

        return spots

    @classmethod
    def from_npz(cls, npz_file):

        data = np.load(npz_file, allow_pickle=True)

        Hs = data['Hs']
        Ks = data['Ks']
        Ls = data['Ls']
        xs = data['xs']
        ys = data['ys']
        zs = data['zs']
        s1_xs = data['s1_xs']
        s1_ys = data['s1_ys']
        s1_zs = data['s1_zs']
        intensities = data['intensities']
        sigmas = data['sigmas']
        resolutions = data['resolutions']
        Fcs = data['Fcs']
        Fos = data['Fos']
        Fos_scaled = data['Fos_scaled']
        Fos_corrected = data['Fos_corrected']

        material = data['material']
        cif_file = data['cif_file']
        output_prefix = data['output_prefix']
        global_scale = data['global_scale']
        excitation_errors = data['excitation_errors']

        spots = []
        for idx in range(len(Hs)):
            H = Hs[idx]
            K = Ks[idx]
            L = Ls[idx]
            x = xs[idx]
            y = ys[idx]
            z = zs[idx]
            s1_x = s1_xs[idx]
            s1_y = s1_ys[idx]
            s1_z = s1_zs[idx]
            intensity = intensities[idx]
            sigma = sigmas[idx]
            resolution = resolutions[idx]
            excitation_error = excitation_errors[idx]

            Fc = Fcs[idx]
            Fo = Fos[idx]
            Fo_scaled = Fos_scaled[idx]
            Fo_corrected = Fos_corrected[idx]

            spot = Spot(H=H, K=K, L=L, intensity=intensity,
                        sigma=sigma, x=x, y=y, z=z,
                        s1_x=s1_x, s1_y=s1_y, s1_z=s1_z,
                        resolution=resolution,
                        Fc=Fc, Fo_corrected=Fo_corrected,
                        excitation_error=excitation_error)
            spot.Fo = Fo
            spot.Fo_scaled = Fo_scaled

            spots.append(spot)

        slist = cls(spots, material=material, output_prefix=output_prefix)

        slist.cif_file = cif_file
        slist.global_scale = global_scale
        slist.output_prefix = output_prefix

        return slist

    def to_npz(self, npz_file=None):

        if not npz_file:
            npz_file = f"{self.output_prefix}.npz"

        Hs = []
        Ks = []
        Ls = []
        xs = []
        ys = []
        zs = []
        s1_xs = []
        s1_ys = []
        s1_zs = []
        intensities = []
        sigmas = []
        resolutions = []
        Fcs = []
        Fos = []
        Fos_scaled = []
        Fos_corrected = []
        excitation_errors = []

        for spot in self.spots:
            Hs.append(spot.H)
            Ks.append(spot.K)
            Ls.append(spot.L)
            xs.append(spot.x)
            ys.append(spot.y)
            zs.append(spot.z)
            s1_xs.append(spot.s1_x)
            s1_ys.append(spot.s1_y)
            s1_zs.append(spot.s1_z)
            intensities.append(spot.intensity)
            sigmas.append(spot.sigma)
            resolutions.append(spot.resolution)
            Fcs.append(spot.Fc)
            Fos.append(spot.Fo)
            Fos_scaled.append(spot.Fo_scaled)
            Fos_corrected.append(spot.Fo_corrected)
            excitation_errors.append(spot.excitation_error)

        print(f"Saving SpotsList into npz file: {npz_file}")
        np.savez(npz_file, Hs=Hs, Ks=Ks, Ls=Ls, xs=xs, ys=ys, zs=zs,
                 s1_xs=s1_xs, s1_ys=s1_ys, s1_zs=s1_zs,
                 intensities=intensities, sigmas=sigmas,
                 resolutions=resolutions, Fcs=Fcs, Fos=Fos,
                 Fos_scaled=Fos_scaled, Fos_corrected=Fos_corrected,
                 material=self.material,
                 cif_file=self.cif_file,
                 output_prefix=self.output_prefix,
                 excitation_errors=excitation_errors,
                 global_scale=self.global_scale)

    @classmethod
    def from_mtz(cls,
                 mtz_file: str,
                 material: str = 'paracetamol',
                 output_prefix: str = '0000',
                 fitted_profile: bool = True):

        mtz = gemmi.read_mtz_file(mtz_file)

        # Extract Miller indices and data
        H, K, L = (mtz.column_with_label('H').array,
                   mtz.column_with_label('K').array,
                   mtz.column_with_label('L').array)

        # Suppose you want the merged intensity (I) and sigma(I)
        intens = mtz.column_with_label('IMEAN').array
        SIGI = mtz.column_with_label('SIGIMEAN').array

#    obs = {}
#    for i in range(len(intens)):
#        hint = int(h[i])
#        kint = int(k[i])
#        lint = int(ll[i])
#        obs[(hint, kint, lint)] = intens[i]
#
#    return obs

        spots = []
        for i in range(len(H)):

            hh = H[i]
            kk = K[i]
            ll = L[i]
            intensity = intens[i]
            sigma = SIGI[i]
            spot = Spot(hh, kk, ll, intensity, sigma=sigma, x=0, y=0, z=0)
            spots.append(spot)

        return cls(spots, material=material, output_prefix=output_prefix)

    @classmethod
    def from_refl(cls,
                  refl_file: str,
                  expt_file: str,
                  material: str = 'paracetamol',
                  output_prefix: str = '0000',
                  fitted_profile: bool = True,
                  exp_id: int = None,
                  scale: bool = False,
                  ):

        refl = flex.reflection_table.from_file(refl_file)

        if exp_id:
            refl = refl.select(refl['id'] == exp_id)

        spots = []

        expt = ExperimentListFactory.from_json_file(expt_file,
                                                    check_format=False)
        hkl_flex = refl["miller_index"]
        hkl_list = [hkl_flex[i] for i in range(len(hkl_flex))]

        if fitted_profile:
            intensities = list(refl["intensity.prf.value"])
            sigmas = list(refl["intensity.prf.variance"])
        else:
            intensities = list(refl["intensity.sum.value"])
            sigmas = list(refl["intensity.sum.variance"])

        if scale:
            intensities = np.array(refl["intensity.scale.value"])
            # scales = np.array(refl["inverse_scale_factor"])
            # intensities = intensities / scales

        vals = list(refl["xyzobs.px.value"])       # list of floats

        for i in range(len(hkl_list)):

            H, K, L = hkl_list[i]
            x, y, z = vals[i]
            s1_x, s1_y, s1_z = compute_s1(x, y, expt)

            intensity = intensities[i]
            sigma = sigmas[i]
            spot = Spot(H, K, L, intensity, sigma=sigma, x=x, y=y, z=int(z),
                        s1_x=s1_x, s1_y=s1_y, s1_z=s1_z)
            spots.append(spot)

        return cls(spots, material=material, output_prefix=output_prefix)

    def get_ys(self, filter_positive=True):

        ys = []
        for spot in self:
            if spot.intensity > 0:
                ys.append(spot.y)

        ys = np.array(ys)
        return ys

    def get_xs(self, filter_positive=True):

        xs = []
        for spot in self:
            if spot.intensity > 0:
                xs.append(spot.x)

        xs = np.array(xs)
        return xs

    def get_fcs(self, filter_positive=True):

        Fcs = []
        for spot in self:
            if spot.intensity > 0:
                Fcs.append(spot.Fc)

        Fcs = np.array(Fcs)
        return Fcs

    def get_iobs(self, filter_positive=True):

        iobs = []
        for spot in self:
            if spot.intensity > 0:
                iobs.append(spot.intensity)

        iobs = np.array(iobs)
        return iobs

    def filter_by_miller(self, miller_index):

        filtered = []
        for spot in self:
            if spot.is_miller(miller_index):
                filtered.append(spot)

        filtered = SpotsList(filtered, material=self.material,
                             output_prefix='filtered')
        return filtered

    def group_by_image(self):

        groups = {}

        for spot in self:

            if spot.z in groups:
                groups[spot.z].append(spot)
            else:
                groups[spot.z] = [spot]

        # Turn each group into an individual SpotsList
        new_groups = {}
        for key in groups:
            temp_spots = groups[key]
            prefix_name = f"{self.output_prefix}" + f'_group_{key:04d}'
            new_spots = SpotsList(temp_spots, material=self.material,
                                  output_prefix=prefix_name)
            new_groups[key] = new_spots

        self.groups = new_groups

        return new_groups

    def compute_R1_per_image(self, a='Fc', b='Fo'):

        indices = []
        image_scales = []
        R1s = []

        for key in self.groups.keys():
            spots_list = self.groups[key]

            image_scale, R1 = compute_R1_for_spots_list(spots_list,
                                                        a=a, b=b)
            indices.append(key)
            image_scales.append(image_scale)
            R1s.append(R1)

        R1s = np.array(R1s)
        image_scales = np.array(image_scales)
        indices = np.array(indices)

        return indices, R1s, image_scales

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
                    intensity = (spot.Fo_corrected * self.global_scale)**2
                # else:
                #     intensity = spot.intensity
                    sig = spot.sigma
                    line = f"{H:4d}{K:4d}{L:4d}  {intensity:12.4f}  "
                    line += f"{sig:10.4f}\n"
                    f.write(line)

    def is_miller_in(self, miller_index):

        H, K, L = miller_index

        for spot in self.spots:
            if spot.is_miller(H, K, L):
                return True
        return False

    def compute_Fo_and_bulk_scale(self):

        Fo = []
        Fc = []
        for spot in self.spots:
            if spot.intensity > 0:
                fo = np.sqrt(spot.intensity)
                spot.Fo = fo
                Fo.append(fo)
                Fc.append(spot.Fc)

        n_start = len(self.spots)
        n_end = len(Fo)
        print(f"Using {n_end} / {n_start} positive spots for scaling")

        Fo = np.array(Fo)
        Fc = np.array(Fc)

        scale, r1_best = find_best_scale(Fo, Fc)

        # scale = Fo.mean() / Fc.mean()
        self.global_scale = scale
        self.average_Fo = Fo.mean()
        self.median_Fo = np.median(Fo)
        print(f' Computed global scale: {scale:.2f}')
        print(' Scaling spots')

        for spot in self.spots:
            if spot.intensity > 0:
                spot.Fo_scaled = spot.Fo / scale
            else:
                spot.Fo_scaled = None

    def compute_fc(self, nproc=20):

        params = []
        for spot in self.spots:
            params.append((self.cif_file, spot))

        print(' Computing the Fc')
        with Pool(processes=nproc) as pool:
            fcs = pool.map(sf_spot_multiprocess, params)

        for i in range(len(fcs)):
            self.spots[i].Fc = fcs[i]

    def compute_resolutions(self):

        resolution_func = get_resolution_calculator(self.cif_file)

        for spot in self.spots:
            spot.resolution = resolution_func(spot.miller)
            # print("COMPUTED", spot.miller, spot.resolution)

    def plot_fc_vs_predicted(self, filename=None):

        true_fc = []
        pred_fc = []

        scale, R1 = compute_R1_for_spots_list(self.spots,
                                              a='Fc', b='Fo_corrected')

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

        print(f"R1 = {R1:.2f}")

        r1_str = f"{R1:.2f}"
        scale_str = f"{scale:.2f}"
        r1_label = r'${\rm R}_1$ = '
        s_label = r'${\rm s}$ = '

        ax.text(0.05, 0.95, r1_label + r1_str,
                va='top', ha='left', transform=ax.transAxes)
        ax.text(0.05, 0.85, s_label + scale_str,
                va='top', ha='left', transform=ax.transAxes)

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


def compute_R1_for_spots_list(spots_list, a='Fc', b='Fo'):

    if len(spots_list) == 0:
        return 0, 0

    Fas = []
    Fbs = []

    for spot in spots_list:

        Fa = getattr(spot, a)
        Fb = getattr(spot, b)

        if Fa is not None and Fb is not None:
            if not np.isnan(Fa) and not np.isnan(Fb):
                Fas.append(Fa)
                Fbs.append(Fb)
    Fas = np.array(Fas)
    Fbs = np.array(Fbs)

    image_scale = np.median(Fas / Fbs)
    R1 = np.sum(np.abs(Fas - image_scale * Fbs)) / np.sum(Fas)

    return image_scale, R1


def double_exp(x, a, b, c, d):
    return a*np.exp(b*x) + c*np.exp(d*x)
