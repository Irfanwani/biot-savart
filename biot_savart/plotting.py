import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
from .biot_savart import parse_coil

# Plotting routines


def plot_fields(
        Bfields,
        box_size,
        start_point,
        vol_resolution,
        which_plane='z',
        level=0,
        num_contours=50):
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
    X = np.linspace(start_point[0],
                    box_size[0] + start_point[0],
                    int(box_size[0] / vol_resolution) + 1)
    Y = np.linspace(start_point[1],
                    box_size[1] + start_point[1],
                    int(box_size[1] / vol_resolution) + 1)
    Z = np.linspace(start_point[2],
                    box_size[2] + start_point[2],
                    int(box_size[2] / vol_resolution) + 1)

    if which_plane == 'x':

        converted_level = np.where(X >= level)
        B_sliced = [Bfields[converted_level[0]
                            [0], :, :, i].T for i in range(3)]
        x_label, y_label = "y", "z"
        x_array, y_array = Y, Z
    elif which_plane == 'y':
        converted_level = np.where(Y >= level)
        B_sliced = [Bfields[:, converted_level[0]
                            [0], :, i].T for i in range(3)]
        x_label, y_label = "x", "z"
        x_array, y_array = X, Z
    else:
        converted_level = np.where(Z >= level)
        B_sliced = [
            Bfields[:, :, converted_level[0][0], i].T for i in range(3)]
        x_label, y_label = "x", "y"
        x_array, y_array = X, Y

    Bmin, Bmax = np.amin(B_sliced), np.amax(B_sliced)

    component_labels = ['x', 'y', 'z']
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
    axes[0].set_ylabel(y_label + " (cm)")

    for i in range(3):
        contours = axes[i].contourf(x_array, y_array, B_sliced[i],
                                    vmin=Bmin, vmax=Bmax,
                                    cmap=cm.magma, levels=num_contours)
        axes[i].set_xlabel(x_label + " (cm)")
        axes[i].set_title(
            "$\\mathcal{B}$" +
            "$_{}$".format(
                component_labels[i]))

    axes[3].set_aspect(20)
    fig.colorbar(contours, cax=axes[3], extend='both')

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
        coil_points = np.array(parse_coil(input_filename))

        ax.plot3D(coil_points[0, :], coil_points[1, :],
                  coil_points[2, :], lw=2)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.tight_layout()
    plt.show()


def plot_quiver(Bfields, box_size, start_point, vol_resolution):

    # filled contour plot of Bx, By, and Bz on a chosen slice plane
    x = np.linspace(start_point[0],
                    box_size[0] + start_point[0],
                    int(box_size[0] / vol_resolution) + 1)
    y = np.linspace(start_point[1],
                    box_size[1] + start_point[1],
                    int(box_size[1] / vol_resolution) + 1)
    z = np.linspace(start_point[2],
                    box_size[2] + start_point[2],
                    int(box_size[2] / vol_resolution) + 1)

    X, Y, Z = np.meshgrid(x, y, z)

    u = Bfields[:, 0, :, 0]
    v = Bfields[:, 0, :, 1]
    w = Bfields[:, 0, :, 2]

    ax = plt.figure().add_subplot(projection='3d')

    mags = np.linalg.norm(Bfields, axis=3)

    q = ax.quiver(X, Y, Z, u, v, w, cmap="coolwarm")
    q.set_array(mags.reshape(np.prod(mags.shape)))  # Ravel the array

    plt.show()
