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
import timeit

class BasicSetUp(unittest.TestCase):
    def setUp(self):
        self.ManyDiceIndex = 5
        #Uniform
        self.Uniform = [{'Instance': Dice("\\1d4")},
                        {'Instance': Dice("\\3d6")},
                        {'Instance': Dice("4d8")},
                        {'Instance': Dice("\\6d10:<3")},
                        {'Instance': Dice("\\10d6:>4")},
                        {'Instance': Dice("\\1000000d20")}]
        #Normal
        self.Normal = [{'Instance': Dice("\\1d4~n(2)"), 'Mean': 2.0, 'SD': 2.0},
                       {'Instance': Dice("\\3d6~n(-2.5,3.5)"), 'Mean': -2.5, 'SD': 3.5},
                       {'Instance': Dice("4d8~n(3.3)"), 'Mean': 4.0, 'SD': 3.3},
                       {'Instance': Dice("\\6d10:<3~n(4,4.1)"), 'Mean': 4.0, 'SD': 4.1},
                       {'Instance': Dice("\\10d6:>4~n(3.2,5)"), 'Mean': 3.2, 'SD': 5.0},
                       {'Instance': Dice("\\1000000d20~n(10.0)"), 'Mean': 10.0, 'SD': 10.0}]
        #Exponential
        self.Exponential = [{'Instance': Dice("\\1d4~e(0.5)"), 'Lambda': 0.5},
                            {'Instance': Dice("\\3d6~e(0.33)"), 'Lambda': 0.33},
                            {'Instance': Dice("4d8~e(0.25)"), 'Lambda': 0.25},
                            {'Instance': Dice("\\6d10:<3~e(1)"), 'Lambda': 1.0},
                            {'Instance': Dice("\\10d6:>4~e(0.5)"), 'Lambda': 0.5},
                            {'Instance': Dice("\\1000000d20~e(0.1)"), 'Lambda': 0.1}]
        #Rotated Exponential
        self.Rexponential = [{'Instance': Dice("\\1d4~re(0.5)"), 'Lambda': 0.5},
                             {'Instance': Dice("\\3d6~re(0.33)"), 'Lambda': 0.33},
                             {'Instance': Dice("4d8~re(0.25)"), 'Lambda': 0.25},
                             {'Instance': Dice("\\6d10:<3~re(1)"), 'Lambda': 1.0},
                             {'Instance': Dice("\\10d6:>4~re(0.5)"), 'Lambda': 0.5},
                             {'Instance': Dice("\\1000000d20~re(0.1)"), 'Lambda': 0.1}]
        #Extreme Order Probabilities
        self.LowOrder = {'Instance': Dice("\\1000000d6:<10"), 'PickRolls': 10}
        self.HighOrder = {'Instance': Dice("\\1000000d6:>10"), 'PickRolls': 10, 'Faces': 6}
        
        for Distribution in (self.Uniform, self.Normal, self. Exponential, self.Rexponential):
            Distribution[0]['Rolls'], Distribution[0]['Faces'], Distribution[0]['Sum'], Distribution[0]['Descending'], Distribution[0]['PickRolls'] = 1, 4, False, False, 0
            Distribution[1]['Rolls'], Distribution[1]['Faces'], Distribution[1]['Sum'], Distribution[1]['Descending'], Distribution[1]['PickRolls'] = 3, 6, False, False, 0
            Distribution[2]['Rolls'], Distribution[2]['Faces'], Distribution[2]['Sum'], Distribution[2]['Descending'], Distribution[2]['PickRolls'] = 4, 8, True, False, 0
            Distribution[3]['Rolls'], Distribution[3]['Faces'], Distribution[3]['Sum'], Distribution[3]['Descending'], Distribution[3]['PickRolls'] = 6, 10, False, False, 3
            Distribution[4]['Rolls'], Distribution[4]['Faces'], Distribution[4]['Sum'], Distribution[4]['Descending'], Distribution[4]['PickRolls'] = 10, 6, False, True, 4
            Distribution[5]['Rolls'], Distribution[5]['Faces'], Distribution[5]['Sum'], Distribution[5]['Descending'], Distribution[5]['PickRolls'] = 1000000, 20, False, False, 0
        
        for Assignment in ((self.Uniform, Dice.UNIFORM_DIST), (self.Normal, Dice.NORMAL_DIST), (self.Exponential, Dice.EXPONENTIAL_DIST), (self.Rexponential, Dice.ROTATED_EXPONENTIAL_DIST)):
            for TestInstance in Assignment[0]:
                TestInstance['Distribution'] = Assignment[1]
        
class ProperInitialization(BasicSetUp):
    def test_OverallInitialization(self):
        for Distribution in (self.Uniform, self.Normal, self. Exponential, self.Rexponential):
            for InstanceDict in Distribution:
                for Key in ('Rolls', 'Faces', 'Sum', 'Descending', 'Distribution'):
                    #__dict__ is the internal representation of an objects' attributes as a dictionary. 
                    #Useful for DRY sometimes, though only works for variables declared directly in object instance, not the class.
                    self.assertEqual(InstanceDict['Instance'].__dict__[Key], InstanceDict[Key])
                
    def test_NormalInitialization(self):
        for InstanceDict in self.Normal:
            self.assertEqual(InstanceDict['Instance'].Mean, InstanceDict['Mean'])
            self.assertEqual(InstanceDict['Instance'].SD, InstanceDict['SD'])

    def test_ExponentialInitialization(self):
        for InstanceDict in self.Exponential:
            self.assertEqual(InstanceDict['Instance'].Lambda, InstanceDict['Lambda'])

    def test_RexponentialInitialization(self):
        for InstanceDict in self.Rexponential:
            self.assertEqual(InstanceDict['Instance'].Lambda, InstanceDict['Lambda'])
            
    def test_ExceptionCases(self):#Wrong string, orders > Rolls
        BadStrings = ["Will not parse.", "-6d10", "6d-10", "0.5d4", "4d0.2", "\\\\10d6", "10d6~n(3,-5)", "10d6~n(3,0)", "1d4~re(0.0)", "1d4~re(-1.2)"]
        for BadString in BadStrings:
            with self.assertRaises(ValueError):
                Dice(BadString)
        with self.assertRaises(ValueError):
            Dice("6d10:<7")
        
class RollsBasicProperties(BasicSetUp):
    def test_BasicRollFormat(self):
        for Distribution in (self.Uniform, self.Normal, self. Exponential, self.Rexponential):
            for InstanceDict in Distribution:
                InstanceDict['GeneratedRolls']=InstanceDict['Instance'].GenerateRolls()
                if InstanceDict['Rolls'] == 1 or InstanceDict['Sum']:
                    self.assertTrue(numpy.isscalar(InstanceDict['GeneratedRolls']))
                else:
                    if InstanceDict['PickRolls']==0:
                        self.assertEqual(InstanceDict['GeneratedRolls'].size, InstanceDict['Rolls'])
                    else:
                        self.assertEqual(InstanceDict['GeneratedRolls'].size, InstanceDict['PickRolls'])
                    self.assertEqual(InstanceDict['GeneratedRolls'].dtype, numpy.dtype('int'))  #Not very pythonic, but class internals are currently coded to work with int anyways
                    
    def test_RollsDomain(self):
        for Distribution in (self.Uniform, self.Normal, self. Exponential, self.Rexponential):
            GeneratedRolls = Distribution[self.ManyDiceIndex]['Instance'].GenerateRolls()
            #Assert that all outcomes in a large sample fall within the proper range
            self.assertEqual((GeneratedRolls[GeneratedRolls<1]).size, 0)
            self.assertEqual((GeneratedRolls[GeneratedRolls>Distribution[self.ManyDiceIndex]['Faces']]).size, 0)
            #Assert that all possible outcomes are represented in a large sample where the underlying distribution
            #is such that all values are practically guaranteed to appear given the sample size.
            PossibleValues = numpy.arange(1, Distribution[self.ManyDiceIndex]['Faces']+1)
            self.assertEqual(numpy.intersect1d(GeneratedRolls, PossibleValues).size, PossibleValues.size)
    
    def test_ExtremeOrderStatistics(self):#Should probably be formalized
        self.assertEqual(self.LowOrder['Instance'].GenerateRolls().sum(), self.LowOrder['PickRolls'])
        self.assertEqual(self.HighOrder['Instance'].GenerateRolls().sum(), self.HighOrder['PickRolls']*self.HighOrder['Faces'])

class DistributionFunctions(unittest.TestCase):
    def test_ConditionalCdfFunction(self):
        CondNormal = {'Mean': 5.0, 'SD': 3.0, 'Faces': 10}
        Cdf = stats.norm.cdf(numpy.arange(1, CondNormal['Faces']+1, dtype=numpy.float), loc = CondNormal['Mean'], scale = CondNormal['SD'])
        CdfMinus1 = stats.norm.cdf(numpy.arange(1, CondNormal['Faces']+1, dtype=numpy.float) -1.0, loc = CondNormal['Mean'], scale = CondNormal['SD'])             
        DiscretePdf = Cdf - CdfMinus1
        CondNormal['Pdf'] = DiscretePdf/numpy.sum(DiscretePdf)
        CondNormal['Cdf'] = numpy.cumsum(CondNormal['Pdf'])

        Args = {'x': numpy.arange(1, 11), 'loc': 5.0, 'scale': 3.0}
        CandidateCdf = CustomDistributions.RangeConditionalCdf(Distribution=stats.norm, Min=0.0, Max=10.0, **Args)
        Difference = numpy.abs(CandidateCdf-CondNormal['Cdf'])
        self.assertEqual(Difference[Difference>1.0e-12].size, 0)
        
    def test_RotatedExponential(self):
        Args = {'x': numpy.arange(1, 11), 'scale': 5.0}
        Exp = CustomDistributions.FromCdfToPdf(CustomDistributions.RangeConditionalCdf(Distribution=stats.expon, Min=0.0, Max=10.0, **Args))
        Args['loc'] = 10.0
        Rexp = CustomDistributions.FromCdfToPdf(CustomDistributions.RangeConditionalCdf(Distribution=CustomDistributions.RotatedExponential, Min=0.0, Max=10.0, **Args))
        Difference = numpy.abs(Rexp-Exp[::-1])
        self.assertEqual(Difference[Difference>1.0e-12].size, 0)

    def test_Exponential(self):
        TestValues = numpy.array(numpy.arange(0.0, 1.0, 0.01))
        ExponQuantiles = stats.expon.ppf(q=TestValues, scale=1.0)
        CandidateExponQuantiles = CustomDistributions.Exponential.ppf(q=TestValues, scale=1.0)
        Difference = numpy.abs(CandidateExponQuantiles-ExponQuantiles)
        self.assertEqual(Difference[Difference>1.0e-12].size, 0)
        ScipyCallTime = timeit.timeit('stats.expon.ppf(q=TestValues, scale=1.0)', setup ='import scipy.stats as stats;import numpy;TestValues = numpy.array(numpy.arange(0.0, 1.0, 0.01))', number=10000)
        HomeMadeCallTime = timeit.timeit('CustomDistributions.Exponential.ppf(q=TestValues, scale=1.0)', setup = 'import CustomDistributions;import numpy;TestValues = numpy.array(numpy.arange(0.0, 1.0, 0.01))',  number=10000)
        self.assertTrue(HomeMadeCallTime<ScipyCallTime)

if __name__ == '__main__':
    unittest.main()
