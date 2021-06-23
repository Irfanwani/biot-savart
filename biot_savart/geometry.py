import numpy as np
# Tools for making geometrical constructs


def create_B_x_rectangle(
        name, p0=[-21.59, -38.1, -21.59, 1], L=76.20, W=43.18):
    '''
    Creates a rectangle of the Y-Z plane that produces a B_x field.

    name: filename to output to. Should be a .txt file.
    p0: [x0,y0,z0,Current] Starting point of the rectangle.
    L: Length (on Z)
    W: Width (on y)
    '''
    f = open(name, "w")

    p1 = [p0[0], p0[1] + W, p0[2], p0[3]]
    p2 = [p0[0], p0[1] + W, p0[2] + L, p0[3]]
    p3 = [p0[0], p0[1], p0[2] + L, p0[3]]

    line = str(p0)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p1)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p2)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p3)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p0)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)
    f.close()


def create_B_y_rectangle(
        name, p0=[-21.59, -38.1, -21.59, 1], L=76.20, D=43.18):
    '''
    Creates a rectangle of the X-Z plane that produces a B_y field.

    name: filename to output to. Should be a .txt file.
    p0: [x0,y0,z0,Current] Starting point of the rectangle.
    L: Length (on Z)
    D: Depth (on X)
    '''
    f = open(name, "w")

    p1 = [p0[0], p0[1], p0[2] + L, p0[3]]
    p2 = [p0[0] + D, p0[1], p0[2] + L, p0[3]]
    p3 = [p0[0] + D, p0[1], p0[2], p0[3]]

    line = str(p0)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p1)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p2)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p3)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p0)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)
    f.close()


def create_B_z_rectangle(
        name, p0=[-26.67, -26.67, -26.67, 1], H=53.340, DD=53.340):
    '''
    Creates a rectangle of the X-Y plane that produces a B_z field.

    name: filename to output to. Should be a .txt file.
    p0: [x0,y0,z0,Current] Starting point of the rectangle.
    H: Height (on Y)
    DD: Depth (on X)
    '''
    f = open(name, "w")

    p1 = [p0[0] + DD, p0[1], p0[2], p0[3]]
    p2 = [p0[0] + DD, p0[1] + H, p0[2], p0[3]]
    p3 = [p0[0], p0[1] + H, p0[2], p0[3]]

    line = str(p0)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p1)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p2)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p3)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    line = str(p0)
    line = line[1:len(line) - 1] + "\n"
    f.write(line)

    f.close()


def helmholtz_coils(fname1, fname2, numSegments, radius, spacing, current):
    '''
    Creates a pair of Helmholtz Coils that are parallel to the X-Y plane.

    fname1: Name of the file where the first coil will be saved.
    fname2: Name of the file where the second coil will be saved.
    numSegments: Number of segments per coil
    radius: Radius of the coils
    spacing: Spacing between the coils. The first coil will be located at -spacing/2 and the 2nd coil will be located at spacing/2 on the Z plane
    current: The current that goest through each coil.
    '''
    f = open(fname1, "w")
    line = ""
    for i in range(0, numSegments, 1):
        line = str(np.cos(2 * np.pi * (i) / (numSegments - 1)) * radius) + "," + str(np.sin(2 * np.pi *
                                                                                            (i) / (numSegments - 1)) * radius) + "," + str(-spacing / 2.0) + "," + str(current) + "\n"
        f.write(line)
    f.close()

    f = open(fname2, "w")
    line = ""
    for i in range(0, numSegments, 1):
        line = str(np.cos(2 * np.pi * (i) / (numSegments - 1)) * radius) + "," + str(np.sin(2 * np.pi *
                                                                                            (i) / (numSegments - 1)) * radius) + "," + str(spacing / 2.0) + "," + str(current) + "\n"
        f.write(line)
    f.close()


def maxwell_coils(fname1, fname2, fname3, numSegments, radius, current):
    '''
    Creates a set of Maxwell coils that are parallel to the X-Y plane.

    fname1: Name of the file where the first coil will be saved.
    fname2: Name of the file where the second coil will be saved.
    fname3: Name of the file where the third coil will be saved.
    numSegments: Number of segments per coil
    radius: Radius of the coils
    '''
    f = open(fname1, "w")
    line = ""
    for i in range(0, numSegments, 1):
        line = str(np.cos(2 * np.pi * (i) / (numSegments - 1)) * radius * np.sqrt(4/7)) + "," + str(np.sin(2 * np.pi *
                                                                                            (i) / (numSegments - 1)) * radius * np.sqrt(4/7)) + "," + str(-radius * np.sqrt(3/7) /2) + "," + str(current) + "\n"
        f.write(line)
    f.close()

    f = open(fname2, "w")
    line = ""
    for i in range(0, numSegments, 1):
        line = str(np.cos(2 * np.pi * (i) / (numSegments - 1)) * radius * np.sqrt(4/7)) + "," + str(np.sin(2 * np.pi *
                                                                                            (i) / (numSegments - 1)) * radius * np.sqrt(4/7)) + "," + str(radius * np.sqrt(3/7)/2) + "," + str(current) + "\n"
        f.write(line)
    f.close()

    f = open(fname3, "w")
    line = ""
    for i in range(0, numSegments, 1):
        line = str(np.cos(2 * np.pi * (i) / (numSegments - 1)) * radius) + "," + str(np.sin(2 * np.pi *
                                                                                            (i) / (numSegments - 1)) * radius) + "," + str(0) + "," + str(current) + "\n"
        f.write(line)
    f.close()


def create_Bx_circle(fname, numSegments, radius, spacing, current, center):
    '''
    Creates a coil on the Y-Z plane that produces a B_x field.

    fname: Name of the file where the first coil will be saved.
    numSegments: Number of segments per coil
    radius: Radius of the coil
    spacing: Spacing between the coil and the Y-Z plane
    current: The current that goest through the coil.
    center: (y,z) The center of the coil on the Y-Z plane
    '''
    f = open(fname, "w")
    line = ""
    for i in range(0, numSegments, 1):
        line = str(spacing) + "," + str(np.cos(2 * np.pi * (i) / (numSegments - 1)) * radius + center[0]) + "," + str(
            np.sin(2 * np.pi * (i) / (numSegments - 1)) * radius + center[1]) + "," + str(current) + "\n"
        f.write(line)
    f.close()


def create_By_circle(fname, numSegments, radius, spacing, current, center):
    '''
    Creates a coil on the X-Z plane that produces a B_y field.

    fname: Name of the file where the first coil will be saved.
    numSegments: Number of segments per coil
    radius: Radius of the coil
    spacing: Spacing between the coil and the X-Z plane
    current: The current that goest through the coil.
    center: (x,z) The center of the coil on the X-Z plane
    '''
    f = open(fname, "w")
    line = ""
    for i in range(0, numSegments, 1):
        line = str(np.cos(2 * np.pi * (i) / (numSegments - 1)) * radius + center[0]) + "," + str(spacing) + "," + str(
            np.sin(2 * np.pi * (i) / (numSegments - 1)) * radius + center[1]) + "," + str(current) + "\n"
        f.write(line)
    f.close()


def create_Bz_circle(fname, numSegments, radius, spacing, current, center):
    '''
    Creates a coil on the X-Y plane that produces a B_z field.

    fname: Name of the file where the first coil will be saved.
    numSegments: Number of segments per coil
    radius: Radius of the coil
    spacing: Spacing between the coil and the X-Y plane
    current: The current that goest through the coil.
    center: (x,y) The center of the coil on the X-Y plane
    '''
    f = open(fname, "w")
    line = ""
    for i in range(0, numSegments, 1):
        line = str(np.cos(2 * np.pi * (i) / (numSegments - 1)) * radius + center[0]) + "," + str(np.sin(
            2 * np.pi * (i) / (numSegments - 1)) * radius + center[1]) + "," + str(spacing) + "," + str(current) + "\n"
        f.write(line)
    f.close()


def create_Bz_solenoid(fname, numSegments, radius, current, center):
    '''
    Creates a coil on the X-Y plane that produces a B_z field.

    fname: Name of the file where the first coil will be saved.
    numSegments: Number of segments per coil
    radius: Radius of the coil
    spacing: Spacing between the coil and the X-Y plane
    current: The current that goest through the coil.
    center: (x,y) The center of the coil on the X-Y plane
    '''
    f = open(fname, "w")

    windings = 500

    line = ""
    for i in range(0, numSegments, 1):
        line = str(np.cos(2 * np.pi * windings * (i) / (numSegments - 1)) * radius + center[0]) + "," + str(np.sin(
            2 * np.pi * windings * (i) / (numSegments - 1)) * radius + center[1]) + "," + str(20 * i / numSegments) + "," + str(current) + "\n"
        f.write(line)
    f.close()
