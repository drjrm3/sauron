#!/usr/bin/env python3
"""
Saruman: The protein ring finder.

Given 2 zstacks in .tif format containing red and green signals, saruman
takes one of the slices and finds 2D droplets (referred to as 'cells' in the
code) by segmentation and QA, then hands off the cell to the 'grima' function to
perform an analysis. The analysis consists of radially averaging the red and
green signals of the cell and then comparing the signals in one dimension to 
assess the radial localiziation of each signal.

"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
# pylint: disable=wrong-import-position
import matplotlib.pylab as plt

import tifffile as tiff

# We know that when red_min == red_max the zslice will be nan, so we suppress
# this warning.
np.seterr(divide='ignore', invalid='ignore')

NPXL_TRIM = 6
MAX_CELLS_IN_ZSLICE = 999
MAX_CELL_SEARCH_ITERS = 25
TOL_R = 0.18
TOL_G = 0.1
TOL_BW = (TOL_R + TOL_G)/2.0
QUALITY_MIN = 0.05
NTHETA1 = 25
DR1 = 0.01
NTHETA2 = 30
DR2 = 0.1
OUTPUT_IMG_TYPE = "png"
IMSHOW_INTERP = "bilinear"

def get_imgs(red_file, green_file):
    """ Read the tiff files and trim off the borders.

    Input:
        red_file (str): A Z-stack in .tif format containing the Red signal.
        green_file (str): A Z-stack in .tif format containing the Green signal.

    Returns:
        red_img (np.ndarray): A z-dimensional stack of 2D red signals as
            uint16 values.
        green_img (np.ndarray): A z-dimensional stack of 2D green signals as
            uint16 values.
    """

    red_img0 = tiff.imread(red_file)
    green_img0 = tiff.imread(green_file)

    (_, nx0, ny0) = red_img0.shape

    # Trim off the border pixels for each zslice
    red_img = red_img0[:, NPXL_TRIM:nx0-NPXL_TRIM, NPXL_TRIM:ny0-NPXL_TRIM]
    green_img = green_img0[:, NPXL_TRIM:nx0-NPXL_TRIM, NPXL_TRIM:ny0-NPXL_TRIM]

    return (red_img, green_img)

def read_stack(red_file, green_file):
    """ Generate a Zstack from two different signal files.

    Input:
        red_file (str): A Z-stack in .tif format containing the Red signal.
        green_file (str): A Z-stack in .tif format containing the Green signal.

    Returns:
        stack (list): A list of 3 numpy arrays of size (nx, ny, 3).
    """

    (red_img, green_img) = get_imgs(red_file, green_file)

    if red_img.shape != green_img.shape:
        print("ERROR: red_img and green_img are not the same size",
              file=sys.stderr)
        sys.exit(1)

    # pylint: disable=invalid-name
    (nz, nx, ny) = red_img.shape

    # Go through and read the stack, normalizing each zslice
    stack = []
    for izslice in range(nz):
        # TODO: Use the 'normalize' function here
        red_zslice = red_img[izslice, :, :].astype('float')
        red_max = np.amax(red_zslice)
        red_min = np.amin(red_zslice)
        red_zslice = (red_zslice - red_min)/(red_max - red_min)

        green_zslice = green_img[izslice, :, :].astype('float')
        green_max = np.amax(green_zslice)
        green_min = np.amin(green_zslice)
        green_zslice = (green_zslice - green_min)/(green_max - green_min)

        stack.append(np.zeros((nx, ny, 3)))
        stack[-1][:, :, 0] = red_zslice
        stack[-1][:, :, 1] = green_zslice

    return stack

def get_args():
    """ Get Saruman args.

    Args:
        None

    Returns:
        args: Argparse args object.
    """

    parser = argparse.ArgumentParser(prog="saruman")

    parser.add_argument('-r', '--red_file', type=str, required=True,
                        help="TIF file containing red signal.")
    parser.add_argument('-g', '--green_file', type=str, required=True,
                        help="TIF file containing green signal.")
    parser.add_argument('-z', '--iz', type=int, required=True,
                        help="Z slice to analyze.")
    parser.add_argument('-t', '--tmpdir', type=str, required=True,
                        help="Pre-existing tmp directory to use.")
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help="Pre-existing output directory to use.")

    args = parser.parse_args()

    # Validate that files exist
    if not os.path.exists(args.red_file):
        print("[ERROR]: red_file '{}' DNE".format(args.red_file),
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.green_file):
        print("[ERROR]: green_file '{}' DNE".format(args.green_file),
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.tmpdir):
        print("[ERROR]: tmpdir '{}' DNE".format(args.tmpdir), file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.outdir):
        print("[ERROR]: outdir '{}' DNE".format(args.outdir), file=sys.stderr)
        sys.exit(1)

    return args

def get_img_name(red_file, green_file):
    """ Return the base image name shared between the red and green files.

    Input:
        red_file (str): TIF file containing red channel.
        green_file (str): TIF file containing green channel.

    Returns:
        img_name (str): String containing characters shared between red and
            green files.
    """

    red_basename = os.path.basename(red_file)
    green_basename = os.path.basename(green_file)
    img_name = ""
    for (char1, char2) in zip(red_basename, green_basename):
        if char1 == char2:
            img_name += char1
        else:
            break

    if not img_name:
        # TODO: Add a test for this
        print("[ERROR]: Input files should share a common prefix.",
              file=sys.stderr)
        sys.exit(1)

    return img_name

def find_center(cell):
    """ Find the center of a cell image

    Input:
        cell (np.ndarray): A small image of size (_nx, _ny, 3) containing
            a single 2D cross-section of a droplet.

    Returns:
        local_xy (list): The local x, y coordinates of the center of the
            cross section.
    """

    cell_bw = (cell[:, :, 0] + cell[:, :, 1])/2.0
    (_ny, _nx) = cell_bw.shape

    local_xy = [-1, -1]

    # Find local X coordinate
    sum_xs = []
    for _ix in range(_nx):
        sum_red = sum(cell[:, _ix, 0])
        sum_green = sum(cell[:, _ix, 1])
        sum_xs.append(sum_red + sum_green)
    sum_all = sum(sum_xs)
    local_xy[0] = 0
    while sum(sum_xs[:local_xy[0]]) < sum_all/2.0:
        local_xy[0] += 1

    # Find local Y coordinate
    sum_ys = []
    for _iy in range(_ny):
        sum_red = sum(cell[_iy, :, 0])
        sum_green = sum(cell[_iy, :, 1])
        sum_ys.append(sum_red + sum_green)
    sum_all = sum(sum_ys)
    local_xy[1] = 0
    while sum(sum_ys[:local_xy[1]]) < sum_all/2.0:
        local_xy[1] += 1

    return local_xy

def find_cell_borders(zslice):
    """ Find the borders of a cell from the zslice

    Input:
        zslice (np.ndarray): An (nx, ny, 3) array of intensities in [0.0, 1.0]

    Returns:
        row_min (int): Lower bound on row.
        row_max (int): Upper bound on row.
        col_min (int): Lower bound on column.
        col_max (int): Upper bound on column.
    """

    (nrows, ncols, _) = zslice.shape
    zslice_bw = (zslice[:, :, 0] + zslice[:, :, 1])/2.0
    indices = np.where(zslice_bw == np.amax(zslice_bw))
    row_idx = int(indices[0][0])
    col_idx = int(indices[1][0])

    # Find min / max row defining this cell
    row_max = row_idx
    while row_max < nrows and (
            zslice[row_max, col_idx, 0] > TOL_R or
            zslice[row_max, col_idx, 1] > TOL_G
        ):
        row_max += 1
    row_min = row_idx
    while row_min > 0 and (
            zslice[row_min, col_idx, 0] > TOL_R or
            zslice[row_min, col_idx, 1] > TOL_G
        ):
        row_min -= 1

    # Find min / max col defining this cell
    col_max = col_idx
    while col_max < ncols and (
            zslice[row_idx, col_max, 0] > TOL_R or
            zslice[row_idx, col_max, 1] > TOL_G
        ):
        col_max += 1
    col_min = col_idx
    while col_min > 0 and (
            zslice[row_idx, col_min, 0] > TOL_R or
            zslice[row_idx, col_min, 1] > TOL_G
        ):
        col_min -= 1

    return (row_min, row_max, col_min, col_max)

def get_cell(zslice, max_cell_search_iters):
    """ Get a single cell from the zslice

    Input:
        zslice (np.ndarray): An (nx, ny, 3) array of intensities in [0.0, 1.0]

    Returns:
        local_xy (list): Two element list defining local (within a segmented
            cell) center.
        global_xy (list): Two element list defining a global center.
        cell (np.ndarray): Segmented cell image from zslice.
        quality (float): Quality of circularity for the cell
        zslice (np.ndarray): Input zslice with 'cell' segmented image
            removed.
    """

    # Sentinal values
    local_xy = [-1, -1]
    global_xy = [-1, -1]
    cell = np.zeros((3, 3, 3))
    quality = -1
    cell_found = False

    iteration = 0
    while not cell_found and iteration < max_cell_search_iters:
        iteration += 1
        if iteration == max_cell_search_iters:
            print("[WARNING] Could not find cell after {} iterations"\
                  .format(max_cell_search_iters))
            local_xy = [-1, -1]
            break

        # Find row, column of max intensity
        (row_min, row_max, col_min, col_max) = find_cell_borders(zslice)

        # Find the enter of the cell
        cell = np.copy(zslice[row_min:row_max, col_min:col_max, :])
        zslice[row_min:row_max, col_min:col_max, :] = 0.0
        zslice[row_min:row_max, col_min:col_max, 2] = 0.0

        # Find the center
        local_xy = find_center(cell)

        # Get radii
        radii = get_radii(local_xy, cell, NTHETA1)

        # Check for circular quality
        quality = np.std(radii)/np.mean(radii)
        if quality < QUALITY_MIN:
            cell_found = True

    global_xy = [local_xy[0] + col_min, local_xy[1] + row_min]

    return (local_xy, global_xy, cell, quality, zslice)

def get_radii(local_xy, cell, ntheta):
    """ Get a list of radii given a differential of theta

    Input:
        local_xy (list): Two element list defining local (within a segmented
            cell) center.
        cell (np.ndarray): The local image of a cell.
        ntheta (int): The number of rays to use.

    Returns:
        radii (list): List of lengths from center to edge of cell.
    """
    radii = []
    thetas = np.linspace(0.0, 2.0*np.pi, ntheta)
    #thetas = np.linspace(0.0, np.pi, ntheta)
    for theta in thetas:
        radius = 0.0
        x_val = local_xy[0] + radius*np.cos(theta)
        y_val = local_xy[1] + radius*np.sin(theta)
        rgb = get_rgb(x_val, y_val, cell)
        while rgb[0] > TOL_R and rgb[1] > TOL_G:
            radius += DR1
            x_val = local_xy[0] + radius*np.cos(theta)
            y_val = local_xy[1] + radius*np.sin(theta)
            rgb = get_rgb(x_val, y_val, cell)

        radii.append(radius)

    return radii

def get_rgb(x_val, y_val, cell):
    """ Find the bilinear interpolant of the image at a real valued point.

    Input:
        x_val (float): The x position in the local image.
        y_val (float): The y position in the local image.
        cell (np.ndarray): The local image of a cell.

    Returns:
        rgb (np.ndarray): The rgb value of the point.
    """

    (_ny, _nx, _) = cell.shape

    x1_val = np.floor(x_val)
    x2_val = np.ceil(x_val)
    if x1_val == x2_val:
        x2_val += 1

    y1_val = np.floor(y_val)
    y2_val = np.ceil(y_val)
    if y1_val == y2_val:
        y2_val += 1

    if x1_val < 0 or x2_val >= _nx or y1_val < 0 or y2_val >= _ny:
        return [-1, -1, -1]

    coef = 1.0/((x2_val-x1_val)*(y2_val-y1_val))
    term1 = cell[int(y1_val), int(x1_val), :]*(x2_val-x_val)*(y2_val-y_val)
    term2 = cell[int(y1_val), int(x2_val), :]*(x_val-x1_val)*(y2_val-y_val)
    term3 = cell[int(y2_val), int(x1_val), :]*(x2_val-x_val)*(y_val-y1_val)
    term4 = cell[int(y2_val), int(x2_val), :]*(x_val-x1_val)*(y_val-y1_val)

    return coef*(term1 + term2 + term3 + term4)

def get_cells(zslice, max_cells_in_zslice):
    """ Get all cells in the given z-stack zslice

    Input:
        zslice (np.ndarray): An (nx, ny, 3) array of intensities in [0.0, 1.0]

    Returns:
        local_xys (list): List of 'local_xy's (2 element lists) defining local
            centers of each cell.
        global_xys (list): List of 'global_xy's (2 element lists) defining
            global centeres of each cell.
        cells (list): List of np.ndarrays defining segmented cells.
        qualities (list): List of floating point numbers defining the quality
            of circularity for each cell.
    """

    local_xys = []
    global_xys = []
    cells = []
    qualities = []

    icell = 0
    while np.amax(zslice[:, :, 0]) > 0.75 and icell < max_cells_in_zslice:
        print("[INFO] Searching for cell {0:2d} ... Fmax = {1:10.5f}"\
              .format(icell, np.amax(zslice[:, :, 0])), file=sys.stderr)
        (local_xy, global_xy, cell, quality, zslice) = get_cell(zslice, MAX_CELL_SEARCH_ITERS)
        print("lx ly: %3i %3i"%(local_xy[0], local_xy[1]))
        print("gx gy: %3i %3i"%(global_xy[0], global_xy[1]))
        if local_xy[0] > 0.0 and local_xy[1] > 0.0:
            local_xys.append(local_xy)
            global_xys.append(global_xy)
            cells.append(cell)
            qualities.append(quality)
        else:
            break
        icell += 1


    return (local_xys, global_xys, cells, qualities)

def normalize(cell0):
    """ Normalize a (:,:,3) np.ndarray object such that all values in each of
        the first two dimensions are in [0.0, 1.0]

    Input:
        cell0 (np.ndarray): A small image of size (_nx, _ny, 3) containing
            a single 2D cross-section of a droplet.

    Returns:
        cell (np.ndarray): A normalize small image of size (_nx, _ny, 3)
            containing a single 2D cross-section of a droplet.
    """
    cell = np.copy(cell0)
    red_min = np.amin(cell0[:, :, 0])
    red_max = np.amax(cell0[:, :, 0])
    green_min = np.amin(cell0[:, :, 1])
    green_max = np.amax(cell0[:, :, 1])
    cell[:, :, 0] = (cell0[:, :, 0] - red_min)/(red_max - red_min)
    cell[:, :, 1] = (cell0[:, :, 1] - green_min)/(green_max - green_min)

    return cell

def grima(ntheta, dr, local_xy, cell0):
    """ grima analyzes the cell and performs radial averaging and standard
        deviations to come up with a view of the red and green signals as
        one dimensional functions.

    Inputs:
        ntheta (int): Theta differential for moviung around the cell.
        dr (float): Radial differential for moving from the center of the cell.
        local_xy (list): Length 2 list of floats denoting the local center of
            the cell.
        cell0 (np.ndarray): A small image of size (_nx, _ny, 3) containing
            a single 2D cross-section of a droplet.

    Returns:
        all_radii (list): A list of lists containing radial grids of
            differential 'dr' determined by starting at the center of the cell
            (determined by 'local_xy') and walking out to the edge. Each
            element in the list is one angular step around the cell.
        all_red_vals (list): A list of lists containing the intensity of the
            red signal for each radial grid in 'all_radii'
        all_green_vals (list): A list of lists containing the intensity of the
            green signal for each radial grid in 'all_radii'
        radii (list): The longest radial grid in 'all_radii'
        red_avgs (list): A list of the radially averaged red intensities.
        red_stds (list): A list of the standard deviations for each group of
            radial intensities.
        green_avgs (list): A list of the radially averaged green intensities.
        green_stds (list): A list of the standard deviations for each group of
            radial intensities.
        cell (np.ndarray): A small (normalized) image of size (_nx, _ny, 3)
            containing a single 2D cross-section of a droplet.
    """

    # Normalize cell
    cell = normalize(cell0)

    all_radii = []
    all_red_vals = []
    all_green_vals = []

    thetas = np.linspace(0.0, 2.0*np.pi, ntheta+1)

    # Start at the center of the cell and walk to the edge of the image, keeping track of the
    # radius along with red and green intensities along the way.
    for theta in thetas[:-1]:
        radius = 0.0
        x_val = local_xy[0] + radius*np.cos(theta)
        y_val = local_xy[1] + radius*np.sin(theta)
        t_radii = []
        t_red_vals = []
        t_green_vals = []
        rgb = get_rgb(x_val, y_val, cell)
        # TODO: Use TOL_R, TOL_G here
        while np.amin(rgb) >= 0.0:
            radius += dr
            x_val = local_xy[0] + radius*np.cos(theta)
            y_val = local_xy[1] + radius*np.sin(theta)
            rgb = get_rgb(x_val, y_val, cell)

            if np.amin(rgb) >= 0.0:
                t_radii.append(radius)
                t_red_vals.append(rgb[0])
                t_green_vals.append(rgb[1])

        all_radii.append(t_radii)
        all_red_vals.append(t_red_vals)
        all_green_vals.append(t_green_vals)

    # Go through and average the red and green intensities for each radial value
    radii = []
    for t_radii in all_radii:
        for radius in t_radii:
            if radius not in radii:
                radii.append(radius)
                radii = sorted(radii)

    # Go through and get red and green avg and std values for each point in all_radii
    red_avgs = []
    red_stds = []
    green_avgs = []
    green_stds = []
    for radius in radii:
        t_red_vals = []
        t_green_vals = []
        # Go through all rays (center of cell to edge of image)
        for (iray, _) in enumerate(all_radii):
            if radius in all_radii[iray]:
                rad_idx = all_radii[iray].index(radius)
                t_red_vals.append(all_red_vals[iray][rad_idx])
                t_green_vals.append(all_green_vals[iray][rad_idx])

        # Get the avgs and stds
        red_avgs.append(np.average(np.array(t_red_vals)))
        red_stds.append(np.std(np.array(t_red_vals)))

        green_avgs.append(np.average(np.array(t_green_vals)))
        green_stds.append(np.std(np.array(t_green_vals)))

    return (all_radii, all_red_vals, all_green_vals, radii, red_avgs,
            red_stds, green_avgs, green_stds, cell)

def get_half_max(radii, intensities):
    """ Get the radius for which the intensity is 0.5

    Inputs:
        radii (list): List of radial points (floats).
        intensities (list): List of intensities at 'radii'.

    Returns:
        rad_half_max (float): Radius at which the intentisities are half max.
    """

    assert len(radii) == len(intensities)
    assert max(intensities) <= 1.0
    assert min(intensities) >= 0.0

    for (irad, _) in enumerate(intensities):
        # TODO: Make sure irad + 1 is accessible in intensities
        if intensities[irad] > 0.5 and intensities[irad + 1] < 0.5:
            x1_val = radii[irad]
            x2_val = radii[irad+1]

            y1_val = intensities[irad]
            y2_val = intensities[irad+1]

            slope = (y2_val - y1_val)/(x2_val - x1_val)

            red_half_max = (0.5-y1_val)/slope + x1_val

            return red_half_max

    print("[ERROR] get_half_max was not able to find the radius of half max",
          file=sys.stderr)
    sys.exit(1)

def get_shell_width(radii, red_signal, green_signal):
    """ Get the width of the region where the green_signal > red_signal

    Input:
        radii (list): A radial grid.
        red_signal (list): The red intensities at each point in 'radii'.
        green_signal (list): The green intesntieis at each point in 'radii'.

    Returns:
        r_beg (float): Radial point of the beginning of the region.
        r_end (float): Radial point of the end of the region.
        intg (float): Integral of (green_signal - red_signal) over
            (r_beg, r_end)
    """

    r_beg = -1.0
    r_end = -1.0
    intg = 0.0

    irad = 1
    while irad < len(radii):
        rad0 = radii[irad-1]
        rad1 = radii[irad]

        red0 = red_signal[irad-1]
        red1 = red_signal[irad]

        green0 = green_signal[irad-1]
        green1 = green_signal[irad]

        dr = (rad1-rad0)

        if r_beg < 0.0 and green0 < red0 and green1 > red1:
            r_beg = (rad0 + rad1)/2.0

        if r_beg > 0.0 and r_end < 0.0:
            intg += (green1 - red1)*dr

        if r_beg > 0.0 and r_end < 0.0 and green0 > red0 and green1 < red1:
            r_end = (rad0 + rad1)/2.0

        irad += 1

    if r_end < r_beg:
        intg = 0.0
        r_beg = -1.0
        r_end = -1.0

    return (r_beg, r_end, intg)

def plot_cell(local_xy, cell0, cell, all_radii, all_red_vals, all_green_vals, radii,
              red_avgs, red_stds, green_avgs, green_stds, figure_name, dpi=300):
    """ Plot the cell along with our analysis of protein intensities

    Input:
        local_xy (list): The local x, y coordinates of the center of the
            cross section.
        cell0 (np.ndarray): A small (unnormalized) image of size (_nx, _ny, 3)
            containing a single 2D cross-section of a droplet.
        cell (np.ndarray): A small (normalized) image of size (_nx, _ny, 3)
            containing a single 2D cross-section of a droplet.
        all_radii (list): A list of lists containing radial grids of
            differential 'dr' determined by starting at the center of the cell
            (determined by 'local_xy') and walking out to the edge. Each
            element in the list is one angular step around the cell.
        all_red_vals (list): A list of lists containing the intensity of the
            red signal for each radial grid in 'all_radii'
        all_green_vals (list): A list of lists containing the intensity of the
            green signal for each radial grid in 'all_radii'
        radii (list): The longest radial grid in 'all_radii'
        red_avgs (list): A list of the radially averaged red intensities.
        red_stds (list): A list of the standard deviations for each group of
            radial intensities.
        green_avgs (list): A list of the radially averaged green intensities.
        green_stds (list): A list of the standard deviations for each group of
            radial intensities.

    Returns:
        None
    """

    plt.figure(figsize=(15, 5))

    # Plot Saruman's view
    plt.subplot(1, 5, 1)
    plt.imshow(cell0, interpolation=IMSHOW_INTERP)
    plt.plot(local_xy[0], local_xy[1], 'wo')
    (_nr, _nc, _) = cell0.shape
    plt.axis([0, _nc-1, 0, _nr-1])
    plt.title('Saruman')

    # Plot Grima's view
    plt.subplot(1, 5, 2)
    plt.imshow(cell, interpolation=IMSHOW_INTERP)
    plt.plot(local_xy[0], local_xy[1], 'wo')
    (_nr, _nc, _) = cell.shape
    plt.axis([0, _nc-1, 0, _nr-1])
    plt.title('Grima')

    # Plot the raw red and green radial intensities
    plt.subplot(1, 5, 3)
    for (irad, _) in enumerate(all_radii):
        plt.plot(all_radii[irad], all_red_vals[irad], 'r')
        plt.plot(all_radii[irad], all_green_vals[irad], 'g')
    plt.axis([0.0, max(radii), 0.0, 1.0])

    # Calculate mean red and green signals +- std's
    red_avg_minus_std = [avg - std for (avg, std) in zip(red_avgs, red_stds)]
    red_avg_plus_std = [avg + std for (avg, std) in zip(red_avgs, red_stds)]
    green_avg_minus_std = [avg - std for (avg, std) in zip(green_avgs, green_stds)]
    green_avg_plus_std = [avg + std for (avg, std) in zip(green_avgs, green_stds)]

    # Plot the averaged intensities with std's
    plt.subplot(1, 5, 4)

    plt.plot(radii, red_avgs, '.-r')
    plt.plot(radii, red_avg_minus_std, 'r--')
    plt.plot(radii, red_avg_plus_std, 'r--')

    plt.plot(radii, green_avgs, '.-g')
    plt.plot(radii, green_avg_minus_std, 'g--')
    plt.plot(radii, green_avg_plus_std, 'g--')

    plt.axis([0.0, max(radii), 0.0, 1.0])

    # Plot the averaged intensities with std's along with green over red ratio
    # 1) Averaged intensities (as above)
    ax1 = plt.subplot(1, 5, 5)

    ax1.plot(radii, red_avgs, 'r')
    ax1.plot(radii, red_avg_minus_std, 'r--')
    ax1.plot(radii, red_avg_plus_std, 'r--')

    ax1.plot(radii, green_avgs, 'g')
    ax1.plot(radii, green_avg_minus_std, 'g--')
    ax1.plot(radii, green_avg_plus_std, 'g--')

    ax1.axis([0.0, max(radii), 0.0, 1.0])
    ax1.yaxis.tick_right()

    # 2) green over red
    ax2 = ax1.twinx()
    green_over_red = \
        [g_val/r_val for (g_val, r_val) in zip(green_avgs, red_avgs)]
    ax2.plot(radii, green_over_red, '.-k')
    ax2.axis([0.0, max(radii), 0.0, 2.0])
    ax1.yaxis.tick_left()

    # Save the image
    plt.savefig(figure_name, dpi=dpi)
    plt.close()

def main():
    """ Main entry point """
    args = get_args()

    print("[INFO] Saruman is searching for the rings.", file=sys.stderr)

    # Read the Z-stack files
    print("[INFO] Reading zstack ... ", end="", file=sys.stderr)
    zstack = read_stack(args.red_file, args.green_file)
    if args.iz < 0 or args.iz > len(zstack):
        print("\n[ERROR]: iz value '{}' is invalid".format(args.iz),
              file=sys.stderr)
        sys.exit(1)
    zslice = zstack[args.iz]
    print("Done")

    # Get the base image name shared between red and green file
    img_name = get_img_name(args.red_file, args.green_file)
    print("[INFO] img_name: {}".format(img_name), file=sys.stderr)

    # Open the tmp file for this stack
    fout = open("{}/{}.{}.csv".format(args.tmpdir, img_name, args.iz), 'w')

    # Save this zslice as an image in the output directory
    plt.imshow(zslice)
    plt.savefig("{}/{}.all.{}.{}".format(args.outdir, img_name, args.iz,
                                         OUTPUT_IMG_TYPE))

    # Get all the cells
    (local_coords, global_coords, cells, qualities) = \
        get_cells(zslice, MAX_CELLS_IN_ZSLICE)

    # Process the cells
    # TODO: Move this into a function!
    for (local_xy, global_xy, cell0, quality) in \
        zip(local_coords, global_coords, cells, qualities):

        (all_radii, all_red_vals, all_green_vals, radii, red_avgs, red_stds,
         green_avgs, green_stds, cell) = grima(NTHETA2, DR2, local_xy, cell0)

        # Get the red and green radii for which the radially averaged
        # intensities are half of the max
        r_half_max_red = get_half_max(radii, red_avgs)
        r_half_max_green = get_half_max(radii, green_avgs)


        (r_beg, r_end, intg) = get_shell_width(radii, red_avgs, green_avgs)

        fout.write("%i, %i, %i, %.10f, %.10f, %.10f, %.10f, %.10f\n"%(
            global_xy[0], global_xy[1], args.iz, r_beg, r_end, intg,
            r_half_max_red, r_half_max_green))

        # Plot the figures
        figure_name = "{}/{}.{}.{}.{}.{}".format(args.outdir, img_name,
                                                 global_xy[0], global_xy[1],
                                                 args.iz, OUTPUT_IMG_TYPE)

        plot_cell(local_xy, cell0, cell, all_radii, all_red_vals, all_green_vals,
                  radii, red_avgs, red_stds, green_avgs, green_stds, figure_name)

    fout.close()
    print("[INFO] Saruman is finished searching for the rings.", file=sys.stderr)



if __name__ == "__main__":
    main()
