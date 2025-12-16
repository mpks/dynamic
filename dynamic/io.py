from dials.array_family import flex
import gemmi
import numpy as np


def fcf_get_iobs_fcal(fcf_file):
    vals = np.loadtxt(fcf_file).T
    h, k, l, Iobs, sig, Fcal, rest = vals

    Iobs_vals = {}
    Fcal_vals = {}
    for i in range(len(h)):
        H = h[i]
        K = k[i]
        L = l[i]
        intensity = Iobs[i]
        fcal = Fcal[i]
        # print('READ', H, K, L, intensity, fcal)
        Iobs_vals[(H, K, L)] = intensity
        Fcal_vals[(H, K, L)] = fcal

    return Iobs_vals, Fcal_vals


def read_mtz_file(mtz_file):

    mtz = gemmi.read_mtz_file(mtz_file)

    # Extract Miller indices and data
    h, k, ll = (mtz.column_with_label('H').array,
                mtz.column_with_label('K').array,
                mtz.column_with_label('L').array)

    # Suppose you want the merged intensity (I) and sigma(I)
    intens = mtz.column_with_label('IMEAN').array
    # SIGI = mtz.column_with_label('SIGIMEAN').array

    obs = {}
    for i in range(len(intens)):
        hint = int(h[i])
        kint = int(k[i])
        lint = int(ll[i])
        obs[(hint, kint, lint)] = intens[i]

    return obs


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


def save_dials_hkl(filename, I_dict, I_sig_dict):

    with open(filename, 'w') as f:

        for hkl in I_dict:
            H, K, L = hkl
            II = I_dict[hkl]
            sigI = I_sig_dict[hkl]

            line = f"{H:4d}{K:4d}{L:4d}  {II:12.4f}  {sigI:10.4f}\n"
            f.write(line)


def read_dials_hkl(hkl_file):

    data = np.loadtxt(hkl_file).T
    H, K, L, I, Isig = data

    I_dict = {}
    Isig_dict = {}
    for i in range(len(H)):
        hh = int(H[i])
        kk = int(K[i])
        ll = int(L[i])
        intensity = float(I[i])
        sig_intensity = float(Isig[i])
        I_dict[(hh, kk, ll)] = intensity
        Isig_dict[(hh, kk, ll)] = sig_intensity

    return I_dict, Isig_dict


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
        x, y, z = val
        obs[(h, k, ll)] = intens
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
