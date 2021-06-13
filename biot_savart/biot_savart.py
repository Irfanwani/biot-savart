'''
Biot-Savart Magnetic Field Calculator v5.0
Mingde Yin
Ryan Zazo

June 2021

All lengths are in cm, B-field is in G
'''

import numpy as np

'''
Feature Wishlist:
    improve plot_coil with different colors for different values of current

    accelerate integrator to use meshgrids directly instead of a layer of for loop

    DONE
    get parse_coil to use vectorized function instead of for loop
'''


def parse_coil(filename) -> "tuple[np.ndarray, np.ndarray]":
    '''
    Parses 4 column CSV into x,y,z,I slices for coil.

    Each (x,y,z,I) entry defines a vertex on the coil.

    The current I of the vertex, defines the amount of
    current running through the next segment of coil, in amperes.

    i.e. (0, 0, 1, 2), (0, 1, 1, 3), (1, 1, 1, 4) means that:
    - There are 2 amps of current running between points 1 and 2
    - There are 3 amps of current running between points 2 and 3
    - The last bit of current is functionally useless.

    Returns coil and current
    '''
    coil_and_current = np.loadtxt(filename, delimiter=",")
    return coil_and_current[:, :3], coil_and_current[:, 3]


def slice_coil(coil, steplength):
    '''
    Slices a coil into pieces of size steplength.

    If the coil is already sliced into pieces smaller than that, this does nothing.
    '''
    def interpolate_points(p1, p2, parts):
        '''
        Produces a series of linearly spaced points between two given points in R3+I

        Linearly interpolates X,Y,Z; but keeps I the SAME

        i.e. (0, 2, 1, 3), (3, 4, 2, 5), parts = 2:
        (0, 2, 1, 3), (1.5, 3, 1.5, 3), (3, 4, 2, 5)
        '''
        return np.column_stack(
            (np.linspace(
                p1[0],
                p2[0],
                parts +
                1),
                np.linspace(
                p1[1],
                p2[1],
                parts +
                1),
                np.linspace(
                p1[2],
                p2[2],
                parts +
                1),
                p1[3] *
                np.ones(
                (parts +
                 1))))

    # fill column with dummy data, we will remove this later.
    newcoil = np.zeros((1, 4))

    segment_starts = coil[:, :-1]
    segment_ends = coil[:, 1:]
    # determine start and end of each segment

    segments = segment_ends - segment_starts
    segment_lengths = np.apply_along_axis(np.linalg.norm, 0, segments)
    # create segments; determine start and end of each segment, as well as
    # segment lengths

    # chop up into smaller bits (elements)

    stepnumbers = (segment_lengths / steplength).astype(int)
    # determine how many steps we must chop each segment into

    for i in range(segments.shape[1]):
        newrows = interpolate_points(
            segment_starts[:, i], segment_ends[:, i], stepnumbers[i])
        # set of new interpolated points to feed in
        newcoil = np.vstack((newcoil, newrows))

    if newcoil.shape[0] % 2 != 0:
        newcoil = np.vstack((newcoil, newcoil[-1, :]))
    # Force the coil to have an even number of segments, for Richardson
    # Extrapolation to work

    return newcoil[1:, :].T  # return non-dummy columns


def calculate_field(coil, x, y, z):
    '''
    Calculates magnetic field vector as a result
    of some position and current x, y, z, I
    [In the same coordinate system as the coil]

    Coil: Input Coil Positions, already
    sub-divided into small pieces using slice_coil
    x, y, z: position in cm

    Output B-field is a 3-D vector in units of G
    '''
    FACTOR = 0.1  # = mu_0 / 4pi when lengths are in cm, and B-field is in G

    def bs_integrate(start, end):
        '''
        Produces tiny segment of magnetic field vector (dB) using the midpoint approximation over some interval

        TODO for future optimization: Get this to work with meshgrids directly
        '''
        dl = (end - start).T
        mid = (start + end) / 2
        position = np.array((x - mid[0], y - mid[1], z - mid[2])).T
        # relative position vector
        mag = np.sqrt((x - mid[0])**2 + (y - mid[1])**2 + (z - mid[2])**2)
        # magnitude of the relative position vector

        return start[3] * np.cross(dl[:3], position) / \
            (mag ** 3)[np.newaxis, :]
        # Apply the Biot-Savart Law to get the differential magnetic field
        # current flowing in this segment is represented by start[3]

    B = 0

    # midpoint integration with 1 layer of Richardson Extrapolation
    starts, mids, ends = coil[:, :-1:2], coil[:, 1::2], coil[:, 2::2]

    # run along axis, then sum along axes

    for start, mid, end in np.nditer([starts, mids, ends], flags=[
                                     'external_loop'], order='F'):
        # use numpy fast indexing
        fullpart = bs_integrate(start, end)  # stage 1 richardson
        halfpart = bs_integrate(start,
                                mid) + bs_integrate(mid,
                                                    end)  # stage 2 richardson

        # richardson extrapolated midpoint rule
        B += 4 / 3 * halfpart - 1 / 3 * fullpart
    # return SUM of all components as 3 (x,y,z) meshgrids for (Bx, By, Bz)
    # component when evaluated using produce_target_volume
    return B * FACTOR


def produce_target_volume(coil, box_size, start_point, vol_resolution):
    '''
    Generates a set of field vector values for each tuple (x, y, z) in the box.
​
    Coil: Input Coil Positions in format specified above, already sub-divided into small pieces
    box_size: (x, y, z) dimensions of the box in cm
    start_point: (x, y, z) = (0, 0, 0) = bottom left corner position of the box
    vol_resolution: Spatial resolution (in cm)
    '''
    x = np.linspace(start_point[0],
                    box_size[0] + start_point[0],
                    int(box_size[0] / vol_resolution) + 1)
    y = np.linspace(start_point[1],
                    box_size[1] + start_point[1],
                    int(box_size[1] / vol_resolution) + 1)
    z = np.linspace(start_point[2],
                    box_size[2] + start_point[2],
                    int(box_size[2] / vol_resolution) + 1)
    # Generate points at regular spacing, incl. end points

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    # NOTE: Requires axes to be flipped in order for meshgrid to have the
    # correct dimensional order

    return calculate_field(coil, X, Y, Z)


def get_field_vector(targetVolume, position, start_point, volume_resolution):
    '''
    Returns the B vector [Bx, By, Bz] components in a generated Target Volume at a given position tuple (x, y, z) in a coordinate system

    start_point: (x, y, z) = (0, 0, 0) = bottom left corner position of the box
    volume_resolution: Division of volumetric meshgrid (generate a point every volume_resolution cm)
    '''
    relativePosition = (
        (np.array(position) -
         np.array(start_point)) /
        volume_resolution).astype(int)
    # adjust to the meshgrid's system

    if (relativePosition < 0).any():
        return ("ERROR: Out of bounds! (negative indices)")

    try:
        return targetVolume[relativePosition[0],
                            relativePosition[1], relativePosition[2], :]
    except BaseException:
        return ("ERROR: Out of bounds!")
    # basic error checking to see if you actually got a correct input/output


'''
- If you are indexing a targetvolume meshgrid on your own, remember to account for the offset (starting point), and spatial resolution
- You will need an index like <relativePosition = ((np.array(position) - np.array(start_point)) / volume_resolution).astype(int)>
'''


def write_target_volume(input_filename, output_filename, box_size, start_point,
                        coil_resolution=1, volume_resolution=1):
    '''
    Takes a coil specified in input_filename, generates a target volume, and saves the generated target volume to output_filename.

    box_size: (x, y, z) dimensions of the box in cm
    start_point: (x, y, z) = (0, 0, 0) = bottom left corner position of the box AKA the offset
    coil_resolution: How long each coil subsegment should be
    volume_resolution: Division of volumetric meshgrid (generate a point every volume_resolution cm)
    '''
    coil = parse_coil(input_filename)
    chopped = slice_coil(coil, coil_resolution)
    targetVolume = produce_target_volume(
        chopped, box_size, start_point, volume_resolution)

    with open(output_filename, "wb") as f:
        np.save(f, targetVolume)
    # stored in standard numpy pickle form


def read_target_volume(filename):
    '''
    Takes the name of a saved target volume and loads the B vector meshgrid.
    Returns None if not found.
    '''
    targetVolume = None

    try:
        with open(filename, "rb") as f:
            targetVolume = np.load(f)
        return targetVolume
    except BaseException:
        pass


if __name__ == "__main__":
    # vol = read_target_volume("solenoid_field.txt")
    # print(np.linalg.norm(vol, axis=3).shape)
    # plot_quiver(vol[::4, ::4, ::4], (10, 10, 10), (-5, -5, -5), 2)

    print(parse_coil("coil.txt"))
