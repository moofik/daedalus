import unittest
from sigmoid import sigmoid


class SigmoidFunctionTestCase(unittest.TestCase):
    def test_sigmoid_works_with_positive_numbers(self):
        posRes = sigmoid(999999)
        self.assertAlmostEqual(1, posRes)

    def test_sigmoid_works_with_negative_numbers(self):
        negRes = sigmoid(-999999)
        self.assertAlmostEqual(0, negRes)

    def test_sigmoid_word_with_zero(self):
        zeroRes = sigmoid(0)
        self.assertEquals(zeroRes, 0.5)



if __name__ == '__main__':
    unittest.main()
