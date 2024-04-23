import numpy as np
from scipy.sparse import diags
from list_swwe_function import linearised_SWWE_matrix_supportive
import unittest

class TestLinearisedSWWEMatrixSupportive(unittest.TestCase):
    def test_linearised_SWWE_matrix_supportive(self):
        n = 5
        delta_x = 0.1
        Q, A, P_inv = linearised_SWWE_matrix_supportive(n, delta_x)
        
        # Test the dimensions of the matrices
        self.assertEqual(Q.shape, (n, n))
        self.assertEqual(A.shape, (n, n))
        self.assertEqual(P_inv.shape, (n, n))
        
        # Test the values of the matrices
        expected_Q = np.array([[-0.5, 0.5, 0.0, 0.0, 0.0],
                                [-0.5, 0.0, 0.5, 0.0, 0.0],
                                [0.0, -0.5, 0.0, 0.5, 0.0],
                                [0.0, 0.0, -0.5, 0.0, 0.5,],
                                [0.0, 0.0, 0.0, -0.5, 0.5]])
        expected_A = np.array([[-1.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, -2.0, 1.0, 0.0, 0.0],
                                [0.0, 1.0, -2.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0, -2.0, 1.0],
                                [0.0, 0.0, 0.0, 1.0, -1.0]])
        expected_P_inv = np.array([[20.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 10.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 10.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 10.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 20.0]])
        
        np.testing.assert_array_equal(Q.toarray(), expected_Q)
        np.testing.assert_array_equal(A.toarray(), expected_A)
        np.testing.assert_array_equal(P_inv.toarray(), expected_P_inv)

if __name__ == '__main__':
    unittest.main()