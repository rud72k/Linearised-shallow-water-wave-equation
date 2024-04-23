import unittest
from flowtype import flowtype_function

class TestFlowType(unittest.TestCase):
    def test_critical_flow(self):
        U = 4.0
        c = 4.0
        expected_type = "critical"
        result = flowtype_function(U, c)
        self.assertEqual(result, expected_type)

    def test_supercritical_flow(self):
        U = 5.0
        c = 3.0
        expected_type = "supercritical"
        result = flowtype_function(U, c)
        self.assertEqual(result, expected_type)

    def test_subcritical_flow(self):
        U = 1.0
        c = 1.44
        expected_type = "subcritical"
        result = flowtype_function(U, c)
        self.assertEqual(result, expected_type)

if __name__ == '__main__':
    unittest.main()