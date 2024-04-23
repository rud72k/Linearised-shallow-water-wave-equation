import unittest
import numpy as np
from list_initial_conditions import init_height_gaussian, init_height_zero, init_height_constant, init_speed_zero, initial_condition

class InitialConditionsTestCase(unittest.TestCase):
    def test_init_height_gaussian(self):
        x_array = np.array([1, 2, 3, 4, 5])
        height = 1.0
        expected_result = np.array([0.01831564, 0.36787944, 1.        , 0.36787944, 0.01831564])
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
        result = init_speed_zero(x_array)
        np.testing.assert_allclose(result, expected_result)

    def test_initial_condition(self):
        x_coord = np.array([1, 2, 3, 4, 5])
        height = 2.0
        speed = 3.0
        initial_type = 1
        expected_result = np.array([[0.03663128, 0.73575888, 2.        , 0.73575888, 0.03663128],
                                   [3,3,3,3,3]])
        result = initial_condition(x_coord, height, speed, initial_type)
        np.testing.assert_allclose(result, expected_result)

if __name__ == '__main__':
    unittest.main()