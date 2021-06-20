'''
Biot-Savart Magnetic Field Calculator v5.0
Mingde Yin
Ryan Zazo

June 2021

All lengths are in cm, B-field is in G
'''

import numpy as np
from .inputs import parse_coil

'''
Feature Wishlist:
    improve plot_coil with different colors for different values of current

    DONE
    accelerate integrator to use meshgrids directly instead of a layer of for loop

    DONE
    get parse_coil to use vectorized function instead of for loop
'''


def slice_coil(coil: np.ndarray, currents: np.ndarray,
               steplength: float) -> "tuple[np.ndarray, np.ndarray]":
    '''
    Slices a coil into pieces of size steplength,
    synchronizing current bits as well.
    If the coil is already sliced into pieces
    smaller than that, this does nothing.
    '''

    def interpolate_points(p1, p2, current, parts):
        '''
        Produces a series of linearly spaced points between two given points,
        with constant current

        e.g. [0, 2, 1], [3, 4, 2], 1, parts=2
        >>> [[0, 2, 1], [1.5, 3, 1.5], [3, 4, 2]], [1, 1, 1]
        '''
        coil_points = np.linspace(p1, p2, parts + 1)
        currents = current * np.ones(parts + 1)
        # TODO see if this actually works properly

        return coil_points, currents

    # fill column with dummy data, we will remove this later.
    newcoil = np.zeros((1, 3))
    newcurrents = np.zeros(1)

    segment_starts = coil[:-1, :]
    segment_ends = coil[1:, :]
    # determine start and end of each segment

    segments = segment_ends - segment_starts
    segment_lengths = np.linalg.norm(segments, axis=1)
    # create segments; determine start and end of each segment,
    # as well as segment lengths

    # chop up into smaller bits (elements)

    stepnumbers = (segment_lengths / steplength).astype(int)
    # determine how many steps we must chop each segment into

    for i in range(segments.shape[0]):
        newrows, newcurrs = interpolate_points(
            segment_starts[i], segment_ends[i], currents[i], stepnumbers[i])
        # set of new interpolated points to feed in
        newcoil = np.vstack((newcoil, newrows))
        newcurrents = np.hstack((newcurrents, newcurrs))

    return newcoil[1:], newcurrents[1:]  # return non-dummy columns


def calculate_field(coil: np.ndarray, current: np.ndarray,
                    positions: np.ndarray) -> np.ndarray:
    '''
    Calculates free space magnetic flux density as a result
    of some position and current x, y, z, I
    [In the same coordinate system as the coil]

    Coil with N segments defined by N+1 points
    Space with U x V x W dimensions

    Coil: Input Coil Positions, already
    sub-divided into small pieces using slice_coil
    Array shape (N, 3)

    Positions: (U x V x W x 3) array of positions in cm

    Output B-field is a 3-D vector in units of G
    '''
    FACTOR = 0.1
    # mu_0 / 4pi when lengths are in cm, and B-field is in G, current in A

    midpoints = (coil[1:] + coil[:-1])/2  # midpoints of each coil position
    dl = np.diff(coil, axis=0)  # dl row vectors for each segment
    R_Rprime = positions[:, :, :, np.newaxis, :] - \
        midpoints[np.newaxis, np.newaxis, np.newaxis, :]
   
    # R - R' term, of shape(U, V, W, N, 3)
    mags = np.linalg.norm(R_Rprime, axis=-1)

    elemental_integrands = FACTOR * \
        current[np.newaxis, np.newaxis, np.newaxis, :-1, np.newaxis] * \
        np.cross(dl, R_Rprime) / mags[:, :, :, :, np.newaxis]**3
    # Evaluate the integrand (U, V, W, N, 3) using BSL
    # BSL is current * mu/4pi * dl x (R-R') / |R-R'|^3
    # The "area" underneath each rectangle

    return np.sum(elemental_integrands, axis=-2)  # sum of all "areas"


def generate_positions(box_size: tuple,
                       start_point: tuple,
                       vol_resolution: float) -> np.ndarray:
    '''
    Generates positions at which magnetic field is evaluated
    Box Size: 3 element tuple of form (u, v, w),
    where u, v, w are the x, y, z dimensions of the box in cm

    Start Point: 3 element tuple of form (a, b, c)
    where a, b, c are the x, y, z coordinates
    of the starting corner of the box in cm

    Vol Resolution: Resolution of the grid in cm.
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
    xx, yy, zz, = np.meshgrid(x, y, z, indexing="ij")

    return np.transpose(np.array([xx, yy, zz]), axes=(1, 2, 3, 0))
    # Reorder axes so that the last axis is the (x, y, z) row vector


def produce_target_volume(coil: np.ndarray, current: np.ndarray,
                          box_size: tuple, start_point: tuple,
                          vol_resolution: float) -> "tuple[np.ndarray, np.ndarray]":
    '''
    Generates a set of field vector values for each tuple (x, y, z) in the box.
​
    Coil, Current: Input Coil Positions and Currents in format specified above,
    already sub-divided into small pieces

    box_size: (x, y, z) dimensions of the box in cm
    start_point: (x, y, z) = (0, 0, 0) = bottom left corner position of the box
    vol_resolution: Spatial resolution (in cm)

    Returns: B fields and Positions of Evaluation
    '''
    positions = generate_positions(box_size, start_point, vol_resolution)
    return calculate_field(coil, current, positions), positions


def get_field_vector(targetVolume, position, start_point, volume_resolution):
    # REDO
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


def write_target_volume(input_filename: str, output_filename: str,
                        box_size: tuple, start_point: tuple,
                        coil_resolution: float = 1,
                        volume_resolution: float = 1) -> None:
    '''
    Takes a coil specified in input_filename, generates a target volume, and saves the generated target volume to output_filename.

    box_size: (x, y, z) dimensions of the box in cm
    start_point: (x, y, z) = (0, 0, 0) = bottom left corner position of the box AKA the offset
    coil_resolution: How long each coil subsegment should be
    volume_resolution: Division of volumetric meshgrid (generate a point every volume_resolution cm)
    '''
    coil, current = slice_coil(*parse_coil(input_filename), coil_resolution)

    fields, positions = produce_target_volume(
        coil, current, box_size, start_point, volume_resolution)

    with open(f"{output_filename}_fields", "wb") as f:
        np.save(f, fields)

    with open(f"{output_filename}_positions", "wb") as f:
        np.save(f, positions)
    # stored in standard numpy pickle form
