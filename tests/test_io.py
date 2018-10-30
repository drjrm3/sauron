#!/usr/bin/env python3
""" Testing suite to validate the ingestion of data """

import unittest
import os
from sauron import saruman

class TestDataImport(unittest.TestCase):
    """ Class to test the import of data """

    def test_get_imgs(self):
        """ Tests the 'get_imgs' function """
        data_dir = os.path.abspath(os.path.join(__file__, '../data'))
        red_file = os.path.join(data_dir, 'NS_10_10_PEG50_Z_ave-8_640.tif')
        green_file = os.path.join(data_dir, 'NS_10_10_PEG50_Z_ave-8_488.tif')

        (red_img, green_img) = saruman.get_imgs(red_file, green_file)
        self.assertEqual(red_img.shape, (53, 500, 500))
        self.assertEqual(green_img.shape, (53, 500, 500))

    def test_read_stack(self):
        """ Tests the 'read_stack' function """
        data_dir = os.path.abspath(os.path.join(__file__, '../data'))
        red_file = os.path.join(data_dir, 'NS_10_10_PEG50_Z_ave-8_640.tif')
        green_file = os.path.join(data_dir, 'NS_10_10_PEG50_Z_ave-8_488.tif')

        stack = saruman.read_stack(red_file, green_file)
        self.assertEqual(len(stack), 53)
        self.assertEqual(stack[0].shape, (500, 500, 3))

    def test_get_img_name(self):
        """ Tests the 'get_img_name' function """

        inputs = [
            ("NS_10_10_PEG50_Z_ave-8_640.tif", "NS_10_10_PEG50_Z_ave-8_488.tif"),
            ("example_foo", "example_bar"),
            ("_matches_foo_matches", "_matches_bar_matches")
        ]

        expected_outputs = [
            "NS_10_10_PEG50_Z_ave-8_",
            "example_",
            "_matches_"
        ]

        for (_input, expected_output) in zip(inputs, expected_outputs):
            output = saruman.get_img_name(_input[0], _input[1])
            self.assertEqual(output, expected_output)
