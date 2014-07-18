"""
Copyright (c) 2014 Eric Vallee <eric_vallee2003@yahoo.ca>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import unittest
import numpy
import scipy.stats as stats
from SciDice import Dice
import SciDice.CustomDistributions as CustomDistributions

class StatisticalSetup(unittest.TestCase):
    def setUp(self):
        self.Uniform = Dice("\\1000000d100", False)
        self.Normal = Dice("\\1000000d100~n(30)", False)
        self.Exponential = Dice("\\1000000d100~e(0.02)", False)
        self.Rexponential = Dice("\\1000000d100~re(0.02)", False)

class GoodnessOfFit(StatisticalSetup):
    def ChiSquareTest(self, Distribution, PValue):
        ExpectedFrequencies = Distribution.Pdf*float(Distribution.Rolls)
        ObservedFrequencies = numpy.bincount(Distribution.GenerateRolls()-1).astype(numpy.float)
        T = (numpy.square(ObservedFrequencies-ExpectedFrequencies)/ExpectedFrequencies).sum()
        self.assertTrue(stats.chisqprob(T, Distribution.Faces-1)>=PValue)

    def test_UniformChiSquare(self):
        self.ChiSquareTest(self.Uniform, 0.01)

    def test_NormalChiSquare(self):
        self.ChiSquareTest(self.Normal, 0.01)

    def test_ExponentialChiSquare(self):
        self.ChiSquareTest(self.Exponential, 0.01)

    def test_RexponentialChiSquare(self):
        self.ChiSquareTest(self.Rexponential, 0.01)

if __name__ == '__main__':
    unittest.main()
