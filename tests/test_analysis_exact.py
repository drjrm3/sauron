#!/usr/bin/env python3
""" Testing suite to validate the analysis """

import unittest
import os
import numpy as np
from sauron import saruman

class TestCellFinder(unittest.TestCase):
    """ Class to test the analysis of the image """

    def setUp(self):
        data_dir = os.path.abspath(os.path.join(__file__, '../data'))
        red_file = os.path.join(data_dir, 'NS_10_10_PEG50_Z_ave-8_640.tif')
        green_file = os.path.join(data_dir, 'NS_10_10_PEG50_Z_ave-8_488.tif')

        self.stack = saruman.read_stack(red_file, green_file)

        self.iz = 23
        self.row_min = 95
        self.row_max = 164
        self.col_min = 405
        self.col_max = 470

        self.cell = np.copy(
            self.stack[self.iz][
                self.row_min:self.row_max,
                self.col_min:self.col_max, :])

    def test_get_rgb(self):
        """ Tests the 'get_rgb' function """

        x_vals = np.linspace(10, 12, 4)
        y_vals = np.linspace(10, 12, 4)
        expected_rgb_vals = [[[] for iy in range(4)] for ix in range(4)]
        expected_rgb_vals[0][0] = [0.21829, 0.22799, 0.00000]
        expected_rgb_vals[0][1] = [0.22468, 0.26831, 0.00000]
        expected_rgb_vals[0][2] = [0.25098, 0.30971, 0.00000]
        expected_rgb_vals[0][3] = [0.29720, 0.35218, 0.00000]
        expected_rgb_vals[1][0] = [0.23304, 0.27499, 0.00000]
        expected_rgb_vals[1][1] = [0.25582, 0.31500, 0.00000]
        expected_rgb_vals[1][2] = [0.28638, 0.35878, 0.00000]
        expected_rgb_vals[1][3] = [0.32473, 0.40635, 0.00000]
        expected_rgb_vals[2][0] = [0.27778, 0.32260, 0.00000]
        expected_rgb_vals[2][1] = [0.29777, 0.36356, 0.00000]
        expected_rgb_vals[2][2] = [0.32612, 0.40647, 0.00000]
        expected_rgb_vals[2][3] = [0.36283, 0.45132, 0.00000]
        expected_rgb_vals[3][0] = [0.35251, 0.37079, 0.00000]
        expected_rgb_vals[3][1] = [0.35054, 0.41398, 0.00000]
        expected_rgb_vals[3][2] = [0.37021, 0.45276, 0.00000]
        expected_rgb_vals[3][3] = [0.41150, 0.48712, 0.00000]

        for (x_idx, x_val) in enumerate(x_vals):
            for (y_idx, y_val) in enumerate(y_vals):
                rgb_vals = saruman.get_rgb(x_val, y_val, self.cell)
                for i in range(3):
                    rgb_diff = rgb_vals[i] - expected_rgb_vals[x_idx][y_idx][i]
                    self.assertTrue(np.abs(rgb_diff) < 1e-3)


    def test_find_center(self):
        """ Tests the 'find_center' function """

        local_xy = saruman.find_center(self.cell)
        self.assertEqual(local_xy, [33, 35])

    def test_get_radii(self):
        """ Tests the 'get_radii' function """

        local_xy = saruman.find_center(self.cell)
        self.assertEqual(local_xy, [33, 35])

        # When np.pi is used for 'thetas' (1.1.0.1)
        ''' 
        expected_radii = [
            28.89, 28.98, 29.14, 30.03, 30.13, 30.41, 29.86, 29.99,
            29.76, 30.16, 30.32, 30.07, 30.29, 30.38, 30.60, 30.47,
            30.63, 30.56, 30.56, 30.35, 30.57, 31.00, 31.05, 30.51,
            30.79
        ]
        '''

        # When 2.0*np.pi is used for 'thetas' (1.1.0.2)
        expected_radii = [
            28.89, 29.14, 30.13, 29.86, 29.76, 30.32, 30.29, 30.60,
            30.63, 30.56, 30.57, 31.05, 30.79, 31.13, 32.35, 33.82,
            34.69, 33.05, 31.80, 30.88, 30.20, 29.56, 28.67, 28.67,
            28.89
        ]

        radii = saruman.get_radii(local_xy, self.cell, 25)
        for (expected_radius, radius) in zip(expected_radii, radii):
            self.assertTrue(np.abs(radius - expected_radius) < 1e-7)

    def test_get_half_max(self):
        """ Tests the 'get_half_max' function """

        x_vals = np.linspace(0.0, 1.0, 1000)
        y_vals = 1.0 - x_vals*x_vals
        half = saruman.get_half_max(x_vals, y_vals)
        self.assertTrue(np.abs(half - np.sqrt(2.0)/2.0) < 1e-3)

        x_vals = np.linspace(0.0, 1.0, 1000)
        y_vals = 1.0 - x_vals*x_vals*x_vals*x_vals
        half = saruman.get_half_max(x_vals, y_vals)
        self.assertTrue(np.abs(half - (1.0/2.0)**(1/4)) < 1e-3)

    def test_find_cell_borders(self):
        """ Tests the 'find_cell_borders' function """

        zslice = self.stack[self.iz]

        (row_min, row_max, col_min, col_max) = saruman.find_cell_borders(zslice)

        self.assertEqual(row_min, 0)
        self.assertEqual(row_max, 21)
        self.assertEqual(col_min, 0)
        self.assertEqual(col_max, 37)

    def test_get_cell(self):
        """ Tests the 'get_cell' function """

        zslice = self.stack[self.iz]

        (l_coords, g_coords, cell, _, zslice) = saruman.get_cell(zslice, 25)

        expected_local_coords = [33, 35]
        expected_global_coords = [438, 130]

        self.assertEqual(l_coords, expected_local_coords)
        self.assertEqual(g_coords, expected_global_coords)
        self.assertFalse((cell - self.cell).any())

    def test_grima(self):
        """ Tests the 'grima' function """
        pass

    def test_normalize(self):
        """ Tests the 'normalize' function """
        pass

    def test_get_shell_width(self):
        """ Test 'get_shell_width' functionality """

        thetas = np.linspace(0.0, 4.0*np.pi, 10000)
        xvals = np.cos(thetas)
        yvals = np.sin(thetas)

        (th1, th2, intg) = saruman.get_shell_width(thetas, xvals, yvals)

        self.assertEqual(np.round(np.pi/4, decimals=3), np.round(th1, decimals=3))
        self.assertEqual(np.round(np.pi + np.pi/4, decimals=3), np.round(th2, decimals=3))
        self.assertEqual(np.round(2.0*np.sqrt(2), decimals=3), np.round(intg, decimals=3))

    def test_get_cells(self):
        """ Tests the 'get_cells' function """

        zslice = self.stack[self.iz]

        (l_coords, g_coords, _, _) = saruman.get_cells(zslice, 2)

        expected_local_coords = [
            [33, 35],
            #[29, 30] #When using np.pi for 'thetas' (1.1.0.1)
            [32, 31] #When using 2.0*np.pi for 'thetas' (1.1.0.2)
        ]

        expected_global_coords = [
            [438, 130],
            #[295, 457] # When using np.pi
            [91, 102] # When using 2.0*np.pi
        ]

        for (l_xy, expected_l_xy) in zip(l_coords, expected_local_coords):
            self.assertEqual(l_xy, expected_l_xy)

        for (g_xy, expected_g_xy) in zip(g_coords, expected_global_coords):
            self.assertEqual(g_xy, expected_g_xy)
