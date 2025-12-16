#!/home/marko/stfc/dials_build/conda_base/bin/python3
from dials.array_family import flex
from dynamic.kinematic import get_resolution_calculator


class Spot:

    def __init__(self, miller, intensity, isigma=0.0,
                 substance='paracetamol'):

        if substance == 'paracetamol':
            cif_file = '/home/marko/active/dyn/data/our_paracetamol.cif'

        resolution = get_resolution_calculator(cif_file)

        H, K, L = miller
        self.H = H
        self.K = K
        self.L = L
        self.miller = miller
        self.intensity = intensity
        self.Isig = isigma
        self.resolution = resolution(miller)


def get_miller_on_image(zboxes, image_index):

    millers = []

    for miller in zboxes:

        boxes = zboxes[miller]
        # boxes_start = boxes[0::3]
        # boxes_end = boxes[1::3]
        zz = boxes[2::3]

        for z in zz:
            if image_index == int(z):
                millers.append(miller)
                break

        # for zstart, zend in zip(boxes_start, boxes_end):
        #    if image_index >= zstart and image_index <= zend:
        #        millers.append(miller)
        #        break

    return millers


def read_from_refl(refl_file, fitted_profile=True):

    refl = flex.reflection_table.from_file(refl_file)

    hkl_flex = refl["miller_index"]
    hkl_list = [hkl_flex[i] for i in range(len(hkl_flex))]

    if fitted_profile:
        intensity = list(refl["intensity.prf.value"])
    else:
        intensity = list(refl["intensity.sum.value"])
    boxes = list(refl["bbox"])
    vals = list(refl["xyzobs.px.value"])       # list of floats

    obs = {}
    box_vals = {}
    for hkl, intens, box, val in zip(hkl_list,
                                     intensity,
                                     boxes,
                                     vals):
        h, k, ll = hkl
        h = int(h)
        k = int(k)
        ll = int(ll)
        x, y, z = val
        if (h, k, ll) in obs:
            obs[(h, k, ll)].append(intens)
        else:
            obs[(h, k, ll)] = [intens]

        box_start = box[4]
        box_end = box[5]

        if (h, k, ll) in box_vals:
            box_vals[(h, k, ll)].append(box_start)
            box_vals[(h, k, ll)].append(box_end)
            box_vals[(h, k, ll)].append(z)
        else:
            box_vals[(h, k, ll)] = []
            box_vals[(h, k, ll)].append(box_start)
            box_vals[(h, k, ll)].append(box_end)
            box_vals[(h, k, ll)].append(z)

    return obs, box_vals
