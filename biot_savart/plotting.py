import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
import numpy as np
from .inputs import parse_coil

# Plotting routines


def plot_fields(fields: np.ndarray, positions: np.ndarray,
                which_plane='z', level=0, num_contours=50) -> None:
    # Ryan's Magic Code
    '''
    Plots the set of Bfields in the given region, at the specified resolutions.

    Bfields: A 4D array of the Bfield.
    box_size: (x, y, z) dimensions of the box in cm
    start_point: (x, y, z) = (0, 0, 0) = bottom left corner position of the box AKA the offset
    vol_resolution: Division of volumetric meshgrid (generate a point every volume_resolution cm)
    which_plane: Plane to plot on, can be "x", "y" or "z"
    level : The "height" of the plane. For instance the Z = 5 plane would have a level of 5
    num_contours: THe amount of contours on the contour plot.

    '''
    # filled contour plot of Bx, By, and Bz on a chosen slice plane
    X = positions[:, 0, 0, 0]
    Y = positions[0, :, 0, 1]
    Z = positions[0, 0, :, 2]

    if which_plane == 'x':

        converted_level = np.where(X >= level)

        B_sliced = [fields[converted_level[0]
                            [0], :, :, i].T for i in range(3)]
        x_label, y_label = "y", "z"
        x_array, y_array = Y, Z
    elif which_plane == 'y':
        converted_level = np.where(Y >= level)
        B_sliced = [fields[:, converted_level[0]
                            [0], :, i].T for i in range(3)]
        x_label, y_label = "x", "z"
        x_array, y_array = X, Z
    else:
        converted_level = np.where(Z >= level)
        B_sliced = [
            fields[:, :, converted_level[0][0], i].T for i in range(3)]
        x_label, y_label = "x", "y"
        x_array, y_array = X, Y

    Bmin, Bmax = np.amin(B_sliced), np.amax(B_sliced)

    component_labels = ['x', 'y', 'z']
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
    axes[0].set_ylabel(y_label + " (cm)")

    cmap = cm.magma
    normalizer=Normalize(Bmin,Bmax)
    im=cm.ScalarMappable(norm=normalizer, cmap=cmap)

    for i in range(3):
        contours = axes[i].contourf(x_array, y_array, B_sliced[i],
                                    vmin=Bmin, vmax=Bmax,
                                    cmap=cmap, levels=num_contours)
        axes[i].set_xlabel(x_label + " (cm)")
        axes[i].set_title(
            "$\\mathcal{B}$" +
            "$_{}$".format(
                component_labels[i]))

    axes[3].set_aspect(20)
    fig.colorbar(im, cax=axes[3])

    plt.tight_layout()

    plt.show()


def plot_coil(*input_filenames):
    '''
    Plots one or more coils in space.

    input_filenames: Name of the files containing the coils.
    Should be formatted appropriately.
    '''
    fig = plt.figure()
    tick_spacing = 2
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel("$y$ (cm)")
    ax.set_zlabel("$z$ (cm)")

    for input_filename in input_filenames:
        coil_points, _ = parse_coil(input_filename)

        ax.plot3D(coil_points[:, 0], coil_points[:, 1],
                  coil_points[:, 2], lw=2)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.tight_layout()
    plt.show()


def plot_quiver(fields: np.ndarray, positions: np.ndarray) -> None:
    '''
    3D Quiver plot of the thing in a volume
    '''

    ax = plt.figure().add_subplot(projection='3d')

    mags = np.linalg.norm(fields, axis=-1)

    cmap = cm.magma
    normalizer=Normalize(0, np.amax(mags))
    im=cm.ScalarMappable(norm=normalizer, cmap=cmap)

    q = ax.quiver(positions[:,:,:,0], positions[:,:,:,1], positions[:,:,:,2],
                  fields[:,:,:,0], fields[:,:,:,1], fields[:,:,:,2], cmap=cmap, normalize=True)
    q.set_array(np.repeat(mags.ravel(), 3))  # Ravel the array

    plt.colorbar(im)

    plt.show()
