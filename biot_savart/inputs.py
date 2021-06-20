import numpy as np


def parse_coil(filename: str) -> "tuple[np.ndarray, np.ndarray]":
    '''
    Parses 4 column CSV into x,y,z,I slices for coil.

    Each (x,y,z,I) entry defines a vertex on the coil.

    The current I of the vertex, defines the amount of
    current running through the next segment of coil, in amperes.

    i.e. (0, 0, 1, 2), (0, 1, 1, 3), (1, 1, 1, 4) means that:
    - There are 2 amps of current running between points 1 and 2
    - There are 3 amps of current running between points 2 and 3
    - The last bit of current is functionally useless.

    Returns coil and current, with coil points given row-wise
    '''
    coil_and_current = np.loadtxt(filename, delimiter=",")
    return coil_and_current[:, :3], coil_and_current[:, 3]


def read_target_volume(filename: str) -> "tuple[np.ndarray, np.ndarray]":
    '''
    Takes the name of a saved target volume and loads the B vector meshgrid.
    Returns None if not found.
    '''
    try:
        with open(f"{filename}_fields", "rb") as f:
            fields = np.load(f)
        with open(f"{filename}_positions", "rb") as f:
            positions = np.load(f)
        return fields, positions
    except BaseException:
        return None, None
