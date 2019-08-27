import unittest
from sigmoid import sigmoid


class SigmoidFunctionTestCase(unittest.TestCase):
    def test_sigmoid_works_with_positive_numbers(self):
        res = sigmoid(9)
        self.assertGreaterEqual(res, 0.5)
        self.assertLess(res, 1)

    def test_sigmoid_works_with_negative_numbers(self):
        res = sigmoid(-9)
        self.assertLess(res, 0.1)
        self.assertGreater(res, 0)

    def test_sigmoid_word_with_zero(self):
        res = sigmoid(0)
        self.assertAlmostEqual(res, 0.5)


if __name__ == '__main__':
    unittest.main()
