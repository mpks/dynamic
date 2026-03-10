import gemmi


def get_resolution_calculator(cif_file):

    doc = gemmi.cif.read_file(cif_file)
    block = doc.sole_block()

    def get_val(string):
        return float(block.find_value(string).split('(')[0])

#        float(block.find_value("_cell_length_a")),
#        float(block.find_value("_cell_length_b")),
#        float(block.find_value("_cell_length_c")),
#        float(block.find_value("_cell_angle_alpha")),
#        float(block.find_value("_cell_angle_beta")),
#        float(block.find_value("_cell_angle_gamma"))

    uc = gemmi.UnitCell(get_val("_cell_length_a"),
                        get_val("_cell_length_b"),
                        get_val("_cell_length_c"),
                        get_val("_cell_angle_alpha"),
                        get_val("_cell_angle_beta"),
                        get_val("_cell_angle_gamma")
                        )

    d = uc.calculate_d

    return d


class StructureFactorCalculator:

    def __init__(self, cif_file):

        self.cif_file = cif_file

    def structure_factor(self, miller_index):

        # small = gemmi.read_small_structure('867818.cif')
        # small.change_occupancies_to_crystallographic()
        # calc_x = gemmi.StructureFactorCalculatorX(small.cell)
        # a = calc_x.calculate_sf_from_small_structure(small, miller_index)

        # st = gemmi.read_structure('867818.cif')
        st = gemmi.read_small_structure(self.cif_file)
        st.change_occupancies_to_crystallographic()
        calc_e = gemmi.StructureFactorCalculatorE(st.cell)
        a = calc_e.calculate_sf_from_small_structure(st, miller_index)

        return a


def compute_resolution(cell, h, k, ll):
    d_spacing = cell.calculate_d([h, k, ll])
    return d_spacing


def afloat(value):

    return float(value.split('(')[0])


def load_unit_cell_from_cif(filename):
    doc = gemmi.cif.read_file(filename)
    block = doc.sole_block()

    # Extract unit cell parameters from CIF tags
    a = afloat(block.find_value('_cell_length_a'))
    b = afloat(block.find_value('_cell_length_b'))
    c = afloat(block.find_value('_cell_length_c'))
    alpha = afloat(block.find_value('_cell_angle_alpha'))
    beta = afloat(block.find_value('_cell_angle_beta'))
    gamma = afloat(block.find_value('_cell_angle_gamma'))

    # Create unit cell object
    return gemmi.UnitCell(a, b, c, alpha, beta, gamma)
