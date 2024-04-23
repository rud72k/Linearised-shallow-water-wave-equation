import unittest
import numpy as np
from math import inf
from list_ibvp_solution import linearSWWE_mms, advecting_wave_1, gaussian_wave_1, step_function_wave, zero_wave

class TestListIBVPSolution(unittest.TestCase):
    def test_linearSWWE_mms(self):
        x_array = np.array([1, 2, 3, 4, 5])
        time = 0.25
        H = 1.0
        U = 1.0
        g = 9.81
        phase_constant_1 = 0.0
        phase_constant_2 = 0.0
        Omega_1 = 3.0
        Omega_2 = 2.0
        expected_height = np.array([0.0,0.0,0.0,0.0,0.0])
        expected_speed = np.array([-1.0,1.0,-1.0,1.0,-1.0])
        expected_force = np.array([[2*np.pi,0.0,-2*np.pi,0.0,2*np.pi],
                                   [0.0,0.0,0.0,0.0,0.0]])
        quantity, force = linearSWWE_mms(x_array, time, H, U, g)
        height, speed = quantity
        np.testing.assert_allclose(height, expected_height, atol=1e-8)
        np.testing.assert_allclose(speed, expected_speed, atol=1e-8)
        np.testing.assert_allclose(force, expected_force, atol=1e-8)

    def test_advecting_wave_1(self):
        x_array = np.array([1, 2, 3, 4, 5])
        time_local = 0.25
        H = 1.0
        U = 2.0
        g = 9.81
        some_function = gaussian_wave_1
        expected_result = np.array([[8.83098929e-04, 3.26068151e-02, 5.67493525e-01, 9.83048444e-01, 3.36989126e-01],
                                    [2.76594705e-03, 1.02127543e-01, 1.77744190e+00, 3.07899812e+00, 1.05548093e+00]])
        result = advecting_wave_1(x_array, time_local, H, U, g, some_function)
        np.testing.assert_allclose(result, expected_result)

    def test_gaussian_wave_1(self):
        t = 0.25
        expected_result = 0.24999999999999992
        result = gaussian_wave_1(t)
        self.assertAlmostEqual(result, expected_result)

    def test_step_function_wave(self):
        t = 0.5
        expected_result = 1.0
        result = step_function_wave(t)
        self.assertAlmostEqual(result, expected_result)

    def test_step_function_wave2(self):
        t = 10.5
        expected_result = 0.0
        result = step_function_wave(t)
        self.assertAlmostEqual(result, expected_result)

    def test_zero_wave(self):
        t = 0.5
        expected_result = 0.0
        result = zero_wave(t)
        self.assertAlmostEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()