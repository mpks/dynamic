import numpy as np
from dynamic.spots import Spot, SpotsList


def extract_scan(shelx_spots, file_template='spots_image_####.npz',
                 index_start=0, index_end=1,
                 out_file='rocking_curves.npz'):

    rocking_curves = {}
    npts = abs(index_end - index_start)
    for idx in range(index_start, index_end):

        print('INDEX', idx, index_end - index_start)

        in_file = file_template.replace('####', f"{idx:04d}")
        data = np.load(in_file)

        millers = data['miller']
        ints = data['intensity']

        for ind_m, (miller, intensity) in enumerate(zip(millers, ints)):

            for spot in shelx_spots:
                if spot.is_miller(miller):
                    if spot.miller in rocking_curves:
                        rocking_curves[spot.miller][idx] = intensity
                    else:
                        rocking_curves[spot.miller] = np.zeros(npts)
                        rocking_curves[spot.miller][idx] = intensity

    millers_end = [i for i in rocking_curves.keys()]
    rcs = []

    for mind in millers_end:
        rcs.append(rocking_curves[mind])

    np.savez(out_file, millers=millers_end, rcs=rcs)


def get_shelx_spots(material='paracetamol', exp_id=None):

    if material == 'paracetamol':
        path = '/home/marko/stfc/paracetamol/filtered_one_deg/cluster_01/'
        path += 'dials.hkl_temp'
    else:
        raise ValueError('Unknown material.')

    data = np.loadtxt(path).T

    H, K, L, I, sigma, folder, x, y, z, z1 = data

    spots = []
    for i in range(len(H)):

        spot = Spot(H=int(H[i]), K=int(K[i]), L=int(L[i]),
                    intensity=I[i], sigma=sigma[i], x=x[i], y=y[i],
                    z=int(z[i]))
        spot.zz = float(z[i])
        spot.folder = folder[i]
        if exp_id:
            if spot.folder == exp_id:
                spots.append(spot)
        else:
            spots.append(spot)

    spots = SpotsList(spots, output_prefix='scaled')

    return spots
