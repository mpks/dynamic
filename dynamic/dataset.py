from scitbx import matrix
from itertools import tee
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory as ExpList
from dynamic.spots import SpotsList
import numpy as np


class Dataset:

    def __init__(self, expt_file=None, refl_file=None, images=None):

        self.expt_file = expt_file
        self.refl_file = refl_file
        self.images = images              # List of image objects
        self.exps = ExpList.from_json_file(expt_file, check_format=False)
        self.refs = flex.reflection_table.from_file(refl_file)

    def compare_by_miller(self, threshold=50):

        # all_vecs = self.asociate_exp_image_with_orientation()
        print("Starting comparisson")

        datasets = []
        dataset_names = []
        dataset_indices = []
        for iexp, exp in enumerate(self.exps):
            spots = self.load_spots(iexp)
            datasets.append(spots)
            dataset_names.append(exp.imageset.get_path(0))
            dataset_indices.append(iexp)

        n = len(datasets)
        for i in range(n - 2):
            for j in range(i+1, n):
                print_banner = True
                name_1 = dataset_names[i]
                name_2 = dataset_names[j]
                bstr = "------------------------------------\n"
                bstr += f"{i:04d} {name_1}\n"
                bstr += f"{j:04d} {name_2}\n"

                imgs_1 = datasets[i]
                imgs_2 = datasets[j]
                for ii1, ind1 in enumerate(imgs_1.groups):
                    for ii2, ind2 in enumerate(imgs_2.groups):

                        spots_1 = imgs_1.groups[ind1]
                        spots_2 = imgs_2.groups[ind2]

                        count = count_millers(spots_1, spots_2)

                        if count >= threshold:
                            ostr = f"{i:04d}  {j:04d}  {ind1} {ind2}"
                            ostr += f"  count: {count}"
                            if print_banner:
                                print(bstr)
                                print_banner = False
                            print(ostr)

    def load_spots(self, index=2):

        print("Loading spots", index)

        # temp_refs = self.refs.select(self.refs['id'] == index)

        spots = SpotsList.from_refl(self.refl_file,
                                    self.expt_file,
                                    exp_id=index)
        spots.group_by_image()

        return spots

    def compare_orientations(self, exp_id1, img_id1, exp_id2,
                             img_id2, threshold=0.01):
        """Compare full orientation matrices instead of just one vector"""

        # Get UB matrices for specific images
        ub1 = self.get_ub_matrix_at_image(exp_id1, img_id1)
        ub2 = self.get_ub_matrix_at_image(exp_id2, img_id2)

        # Calculate misorientation
        R = ub2 * ub1.inverse()
        qt = R.r3_rotation_matrix_as_unit_quaternion()
        angle, axis = qt.unit_quaternion_as_axis_and_angle(deg=True)
        # print(exp_id1, img_id1, exp_id2, img_id2, angle)

        return abs(angle)  # Return misorientation angle in degrees

    def get_ub_matrix_at_image(self, exp_id, img_id):
        """Get the UB matrix at a specific image"""
        exp = self.exps[exp_id]
        crystal = exp.crystal

        if crystal.num_scan_points > 0:
            U = matrix.sqr(crystal.get_U_at_scan_point(img_id))
            B = matrix.sqr(crystal.get_B_at_scan_point(img_id))
        else:
            U = matrix.sqr(crystal.get_U())
            B = matrix.sqr(crystal.get_B())

        return U * B

    def new_asociate(self):

        n_exp = len(self.exps)

        for i in range(n_exp-2):
            for j in range(i + 1, n_exp):

                exp1 = self.exps[i]
                exp2 = self.exps[j]

                scan1 = exp1.scan
                num_pts1 = scan1.get_num_images() + 1

                scan2 = exp2.scan
                num_pts2 = scan2.get_num_images() + 1

                pth1 = exp1.imageset.get_path(0)
                pth2 = exp2.imageset.get_path(0)

                for img_i in range(num_pts1):
                    for img_j in range(num_pts2):
                        angle = self.compare_orientations(i, img_i, j, img_j)
                        if angle < 2.0:
                            print(img_i+1, pth1)
                            print(img_j+1, pth2)
                            print(angle)
                            print(20*"=")

    def asociate_exp_image_with_orientation(self):

        print("Getting orientation vectors")

        n_exp = len(self.exps)

        indices = []
        orientation_vecs = []
        all_vecs = []
        for exp_id in range(n_exp):
            vects, _ = self.extract_unit_cell_orientations(exp_id)
            all_vecs.append(vects)
            for ivec, vec in enumerate(vects):
                a, b, c = vec
                indices.append((exp_id, ivec))
                dvec = a + b + c
                dnorm = np.linalg.norm(dvec)
                orientation_vecs.append(dvec / dnorm)
                # exp = self.exps[exp_id]
                # print(f"PATH: {exp.imageset.get_path(0)}")
                # print(exp_id, ivec, dvec / dnorm)

        # nvec = len(orientation_vecs)

        # count = 0
        # for i in range(nvec - 2):
        #     for j in range(i+1, nvec):
        #         v1 = orientation_vecs[i]
        #         v2 = orientation_vecs[j]

        #        diff = np.linalg.norm(v1 - v2)
        #        if diff <= 0.001:
        #            exp1, img1 = indices[i]
        #            exp2, img2 = indices[j]
        #            if exp1 != exp2:
        #                count += 1
        #                print("Match", count, i, j, indices[i],
        #                      indices[j], diff)
        return all_vecs

    def extract_unit_cell_orientations(self, index=2):

        exp = self.exps[index]

        crystal = exp.crystal
        beam = exp.beam
        scan = exp.scan
        gonio = exp.goniometer

        # print(f"PATH: {exp.imageset.get_path(0)}")

        num_pts = scan.get_num_images() + 1

        if beam.num_scan_points > 0:
            us0 = []
            for i in range(beam.num_scan_points):
                s0 = matrix.col(beam.get_s0_at_scan_point(i))
                us0.append(s0.normalize())
        else:
            us0 = [matrix.col(beam.get_unit_s0()) for _ in range(num_pts)]

        if gonio.num_scan_points > 0:
            S_mats = [
                matrix.sqr(gonio.get_setting_rotation_at_scan_point(i))
                for i in range(gonio.num_scan_points)
            ]
        else:
            S_mats = [
                matrix.sqr(gonio.get_setting_rotation()) for _ in
                range(num_pts)
            ]

        F_mats = [matrix.sqr(gonio.get_fixed_rotation()) for _ in
                  range(num_pts)]
        start, stop = scan.get_array_range()
        R_mats = []
        axis = matrix.col(gonio.get_rotation_axis_datum())
        for i in range(start, stop + 1):
            phi = scan.get_angle_from_array_index(i, deg=False)
            ang_as_r3 = axis.axis_and_angle_as_r3_rotation_matrix
            R = matrix.sqr(ang_as_r3(phi, deg=False))
            R_mats.append(R)

        if crystal.num_scan_points > 0:
            U_mats = [
                matrix.sqr(crystal.get_U_at_scan_point(i))
                for i in range(crystal.num_scan_points)
            ]
            B_mats = [
                matrix.sqr(crystal.get_B_at_scan_point(i))
                for i in range(crystal.num_scan_points)
            ]
        else:
            U_mats = [matrix.sqr(crystal.get_U()) for _ in range(num_pts)]
            B_mats = [matrix.sqr(crystal.get_B()) for _ in range(num_pts)]

        check = {len(x) for x in (us0, S_mats, F_mats, R_mats, U_mats)}
        assert len(check) == 1

        SRFU = (S * R * F * U for S, R, F, U in
                zip(S_mats, R_mats, F_mats, U_mats))

        U_frames = []
        for U1, U2 in pairwise(SRFU):
            M = U2 * U1.transpose()
            qt = M.r3_rotation_matrix_as_unit_quaternion()
            angle, axis = qt.unit_quaternion_as_axis_and_angle(deg=False)
            M_half = axis.axis_and_angle_as_r3_rotation_matrix(angle/2,
                                                               deg=False)
            U_frames.append(M_half * U1)

        B_frames = []
        for B1, B2 in pairwise(B_mats):
            B_frames.append((B1 + B2) / 2)

        UB_frames = [U * B for U, B in zip(U_frames, B_frames)]

        frac_mats = [m.transpose() for m in UB_frames]

        # Calculate zone axes, which also requires the beam directions
        # at the framecentres
        us0_frames = []
        for d1, d2 in pairwise(us0):
            us0_frames.append(((d1 + d2) / 2).normalize())

        scale = 1   # I added this line (see original func for more detail)
        uc = exp.crystal.get_unit_cell()
        scale = max(uc.parameters()[0:3])

        zone_axes = [frac * (d * scale) for frac, d in
                     zip(frac_mats, us0_frames)]

        orthog_mats = (frac.inverse() for frac in frac_mats)
        h = matrix.col((1, 0, 0))
        k = matrix.col((0, 1, 0))
        l = matrix.col((0, 0, 1))      # noqa: E741
        real_space_axes = [(o * h, o * k, o * l) for o in orthog_mats]

        # Vectors in the crystal frame
        Os = [b.transpose().inverse() for b in B_frames]
        initial_orient = [(o * h, o * k, o * l) for o in Os]

        return real_space_axes, zone_axes, initial_orient


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def count_millers(spots_1, spots_2):

    counter = 0
    for s1 in spots_1:
        for s2 in spots_2:
            if s1.is_miller(s2.miller):
                counter += 1
    return counter
