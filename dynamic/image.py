

class Image:

    def __init__(self, expt_file, expt_index, z, spots=None,
                 a=None, b=None, c=None):

        self.expt_file = expt_file
        self.expt_index = expt_index
        self.z = z            # Image index
        self.spots = spots    # Spots on that image
        self.a = a            # Unit cell vector a in the lab space
        self.b = b            # Unit cell vector b in the lab space
        self.c = c            # Unit cell vector c in the lab space
