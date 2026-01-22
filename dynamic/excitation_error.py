# excitation_error.py
from dials.command_line.frame_orientations import extract_experiment_data
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix
from dials.array_family import flex


class ExcitationErrorCalculator:
    """
    Computes excitation error for DIALS reflections or custom Spot objects.
    Uses the same definition as DIALS/PETS:
        r = UB * hkl
        s1 = r + s0
        excitation_error = |s0| - |s1|
    """

    def __init__(self, experiments_json: str, exp_id: int = 0,
                 refl_file=None):

        self.reflections = flex.reflection_table.from_file(refl_file)
        experiments = ExperimentListFactory.from_json_file(experiments_json)
        self.experiment = experiments[exp_id]

        self.frame_orientations = extract_experiment_data(self.experiment,
                                                          scale=1)

        Ut = matrix.sqr(self.experiment.crystal.get_U()).transpose()
        self.frame_orientations['rotated_missets'] = [
            M * Ut for M in self.frame_orientations["orientations"]
        ]
        # self.rotated_missets = frame_data['rotated_missets']

        self.crystal = self.experiment.crystal
        self.beam = self.experiment.beam

        # s0 vector and inverse wavelength
        self.s0 = matrix.col(self.beam.get_s0())
        self.inv_wl = self.s0.length()

    def ee(self):

        # Get the frame orientation data
        # directions = self.frame_orientations["directions"]
        zone_axes = self.frame_orientations["zone_axes"]
        # real_space_axes = self.frame_orientations["real_space_axes"]
        rotated_missets = self.frame_orientations["rotated_missets"]

        # Get experiment geometry
        scan = self.experiment.scan
        s0 = matrix.col(self.experiment.beam.get_s0())
        inv_wl = s0.length()
        A = matrix.sqr(self.experiment.crystal.get_A())

        _, _, frames = self.reflections["xyzcal.px"].parts()
        scan = self.experiment.scan
        self.virtual_frames = []
        arr_start, arr_end = scan.get_array_range()
        # Loop over the virtual frames
        starts = list(range(arr_start, arr_end, 1))
        self.n_merged = 1
        for start in starts:
            stop = start + self.n_merged
            if stop > arr_end:
                break

            # Look up the orientation data using an index, which is the centre
            # of the virtual frame, offset so that the scan starts from 0
            centre = (start + stop) / 2.0
            index = int(centre) - arr_start
            M = rotated_missets[index]
            # za = zone_axes[index]

            # If the number of merged frames is even, then we need to average
            # this orientation data with that from the previous frame to get
            # the orientation in the centre of the virtual frame
#            if centre - int(centre) == 0.0:
#                Mprev = rotated_missets[index - 1]
#                RR = M * Mprev.transpose()
#                (
#                    angle,
#                    axis,
#                ) = RR.r3_rotation_matrix_as_unit_quaternion().unit_quaternion_as_axis_and_angle(
#                    deg=False
#                )
#                R = axis.axis_and_angle_as_r3_rotation_matrix(angle / 2, deg=False)
#                M = R * Mprev
#
#                za_prev = zone_axes[index - 1]
#                delta = za - za_prev
#                za = za_prev + delta / 2
#
#            # Rescale zone axis to match PETS format
#            za = self._rescale_zone_axis(za)

            # Calculate the excitation error at this orientation
            UB = flex.mat3_double(len(self.reflections), M * A)
            r = UB * self.reflections["miller_index"].as_vec3_double()
            s1 = r + s0
            excitation_err = inv_wl - s1.norms()

            ########################################################
            millers = self.reflections["miller_index"]
            zs = self.reflections["xyzcal.px"].parts()[2]

            for hkl, zval, ee in zip(millers, zs, excitation_err):
                h, k, l = hkl
                print(f"{h:4d} {k:4d} {l:4d}  z={zval:6.2f}  exc_err={ee: .6e}")


    def excitation_error_for_hkl(self, h, k, l, z=None):
        """
        Compute excitation error for a single reflection.

        Parameters
        ----------
        h,k,l : Miller indices
        z     : scan-point / image index (optional)

        Returns
        -------
        excitation_error : float
        """

        # Get UB matrix
        if z is not None and self.crystal.num_scan_points > 0:
            A = matrix.sqr(self.crystal.get_A_at_scan_point(int(z)))
        else:
            A = matrix.sqr(self.crystal.get_A())

        # r = UB * hkl
        hkl = matrix.col((h, k, l))
        r = A * hkl

        # s1 = r + s0
        s1 = r + self.s0

        # excitation error
        return self.inv_wl - s1.length()

    def attach_to_spots_list(self, spots_list):
        """
        Compute excitation errors for all Spot objects in a SpotsList.
        Adds spot.excitation_error attribute.
        """

        for spot in spots_list.spots:

            # If no image index -> undefined
            if spot.z is None:
                spot.excitation_error = None
                continue

            ee = self.excitation_error_for_hkl(
                spot.H, spot.K, spot.L, z=spot.z
            )
            spot.excitation_error = ee

