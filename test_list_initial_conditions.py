import unittest
import numpy as np
from list_initial_conditions import init_height_gaussian, init_height_zero, init_height_constant, init_speed_zero, initial_condition, bottom_bathymetry_flatbed, step_function_bottom

class InitialConditionsTestCase(unittest.TestCase):
    def test_init_height_gaussian(self):
        x_array = np.array([1, 2, 3, 4, 5])
        height = 1.0
        expected_result = height + 0.01*height*np.array([0.01831564, 0.36787944, 1.        , 0.36787944, 0.01831564])
        result = init_height_gaussian(x_array, height)
        np.testing.assert_allclose(result, expected_result)

    def test_init_height_zero(self):
        x_array = np.array([1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 0, 0, 0])
        result = init_height_zero(x_array)
        np.testing.assert_allclose(result, expected_result)

    def test_init_height_constant(self):
        x_array = np.array([1, 2, 3, 4, 5])
        height = 1.0
        expected_result = np.array([1,1,1,1,1])
        result = init_height_constant(x_array, height)
        np.testing.assert_allclose(result, expected_result)

    def test_init_speed_zero(self):
        x_array = np.array([1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 0, 0, 0])
        result = init_speed_zero(x_array, 0)
        np.testing.assert_allclose(result, expected_result)

    def test_initial_condition_zero(self):
        x_coord = np.array([1, 2, 3, 4, 5])
        height = 1.0
        speed = 1.0
        initial_height_type = 0
        initial_speed_type = 1
        expected_result = np.array([[0, 0, 0, 0,0],
                                   [1,1,1,1,1]])
        result = initial_condition(x_coord, height, speed, initial_height_type, initial_speed_type)
        np.testing.assert_allclose(result, expected_result)

    def test_initial_condition_gaussian(self):
        x_coord = np.array([1, 2, 3, 4, 5])
        height = 1.0
        speed = 1.0
        initial_height_type = 1
        initial_speed_type = 1
        expected_result = np.array([[1.00018316, 1.00367879, 1.01,       1.00367879, 1.00018316],
                                    [1,1,1,1,1]])

    def test_bottom_bathymetry_flatbed(self):
        x_array = np.array([1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 0, 0, 0])
        result = bottom_bathymetry_flatbed(x_array)
        np.testing.assert_allclose(result, expected_result)

    def step_function_bottom(self):
        x_array = np.array([1, 2, 3, 4, 5, 6])
        step_up_x_ratio = 0.5
        expected_result = np.array([0, 0, 0, 1, 1, 1])
        result = step_function_bottom(x_array, step_up_x_ratio)
        np.testing.assert_allclose(result, expected_result)

    def init_height_with_step_function_center(self):
        x_array = np.array([1, 2, 3, 4, 5,6,7,8,9,10,11])
        height = 1.0
        expected_result = np.array([0,0,0,0,1,1,1,0,0,0,0])
        result = init_height_with_step_function_center(x_array, height)
        np.testing.assert_allclose(result, expected_result)

if __name__ == '__main__':
    unittest.main()