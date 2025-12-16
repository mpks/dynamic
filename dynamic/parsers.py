import numpy as np


def parse_shelx_output(ltx_file):

    with open(ltx_file, mode='r') as f:

        lines = f.readlines()

    rs = []
    for line in lines:
        if "as input" in line:
            r_temp = float(line.split()[0])
            rs.append(r_temp)
    rs = np.array(rs)
    if len(rs) == 0:
        return -1
    else:
        return rs.min()
