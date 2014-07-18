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
import numpy
import scipy.stats as stats
import re
from SciDice.CustomDistributions import *
        
class Dice(object):
    """
    -------------------------------------------------------------
    |Usage: Instance = Dice(<Pattern>); Instance.GenerateRolls()|
    -------------------------------------------------------------
    |-> <Patterns> is a string that takes the form: [\\]<Rolls>d<Faces>[:<<Amount>|><Amount>][~n(<NormalMean>,<NormalSD>)|~n(<NormalSD>)]
    |
    |-> '\\': If present will keep the results of the dice rolls separate in an array, otherwise will sum them up into a scalar.
    |
    |-> '<Rolls>d<Faces>': <Rolls is an integer indicate the number of dice to use and <Faces> is an integer indicating the upper range of the dice
    |
    |-> ':<<Amount>|><Amount>': If present, Amount is an integer indicating the number of dice to keep. If preceded by ':>', the top <Amount> dice 
    |   will be kept and if preceded by ':<', the bottom <Amount> dice will be kept. Result will be a sorted array.
    |
    |-> By default, each potential die value X has as their pdf the area (X-1,X) under u(0,<Faces>).
    |
    |-> '~n(<NormalMean>,<NormalSD>)|~n(<NormalSD>)':  If present, the cdf used will be a domain-adjusted (ie, 0 to <Faces>) variant of 
    |   n(<NormalMean>,<NormalSD>), with <NormalMean> defaulting to <Faces>/2 if omitted.
    |   
    |   *****************Example****************
    |   #Roll ten 20-sided dice that are normally distributed with mean 10.0 and sd 6.6, return the 3 highest in an array
    |   Best3GroupOf10AttackRolls = Dice(r'\\10d20:>3~n(10.0,6.6)'); Best3GroupOf10AttackRolls.GenerateRolls()
    |   Output: array([18, 17, 14])
    """
    _Positive_simple_double_exp = '\d+(?:[.]\d+)?'
    _Simple_double_exp = '[-]?' + _Positive_simple_double_exp
    _Normal_exp = '(?:~n[(](?:(?P<NormalMean>'+_Simple_double_exp+')[,])?(?P<NormalSD>'+_Positive_simple_double_exp+')[)])'
    _Exp_exp = '(?:~e[(](?P<ExpLambda>'+_Positive_simple_double_exp+')[)])' #Exp_exp :P
    _Rexp_exp = '(?:~re[(](?P<RotExpLambda>'+_Positive_simple_double_exp+')[)])'
    _Distributions_exp = "(?:"+_Normal_exp+"|"+_Exp_exp+"|"+_Rexp_exp+")"
    _DiceRegex = re.compile('^(?P<NoSum>\\\\)?(?P<Rolls>\d+)d(?P<Faces>\d+)(?:[:](?P<Ascending><|>)(?P<HighLowAmount>\d+))?'+_Distributions_exp+'?$')
    UNIFORM_DIST = 0
    NORMAL_DIST = 1
    EXPONENTIAL_DIST = 2
    ROTATED_EXPONENTIAL_DIST = 3
    
    def __init__(self, Input, LightConstructor=True):
        Match = Dice._DiceRegex.match(Input)
        if Match==None:
            raise ValueError("Unparsable constructor string.")
        self.GeneratorString = Input
        self.Rolls = int(Match.group('Rolls'))
        self.Faces = int(Match.group('Faces'))
        self.Sum = (True if Match.group('NoSum')==None else False)
        self.Descending = (True if Match.group('Ascending') != None and Match.group('Ascending') == '>' else False)
        self.HighLowAmount = (int(Match.group('HighLowAmount')) if Match.group('HighLowAmount')!=None else 0)
        if self.HighLowAmount > self.Rolls:
            raise ValueError("Retained rolls cannot be greater than number of rolls.")
        self.UniformGeneratorRange = None
        if Match.group('NormalSD')!=None:
            self.Distribution = self.NORMAL_DIST
            #Empirically, _GenerateRollTrialError was shown to yield the best overall performance.
            #_GenerateRollCdfSearch is faster in cases where NormalSD is large due to the higher number of misses, but the distribution approaches that found in the uniform distribution over the range of interest in those cases so it's unlikely to see much use.
            #_GenerateRollQuantile should theorically be the fastest, but isn't due to the time it takes to compute stats.norm.ppf, probably because the cdf of the normal distribution and it's inverse are not analytically tractable
            self._GenerateRoll = self._GenerateRollTrialError
            #Interestingly, _GenerateRollsQuantile netted up to ~20% speed improvement on my tests with 10000 Rolls or more on d10~n(5,5), but was significantly slow in the lower ranges
            #With d10~n(5,3), _GenerateRollsTrialError was still faster though even with 10000000 rolls due to fewer misses.
            #_GenerateRollsTrialError is more consistant I find and better in usual scenarios (smaller SD and/or smaller number of rolls)
            self._GenerateRolls = self._GenerateRollsTrialError
            self.SD = float(Match.group('NormalSD'))
            if self.SD==0.0:
                raise ValueError("Standard deviation cannot be zero.")
            self.Mean = (float(self.Faces)/2.0 if Match.group('NormalMean') == None else float(Match.group('NormalMean')))
            self.UniformGeneratorRange = stats.norm.cdf(numpy.array([0.0, float(self.Faces)]), loc=self.Mean, scale=self.SD)
        elif Match.group('ExpLambda')!=None:
            self.Distribution = self.EXPONENTIAL_DIST
            self._GenerateRoll = self._GenerateRollTrialError
            self._GenerateRolls = self._GenerateRollsTrialError
            self.Lambda = float(Match.group('ExpLambda'))
            if self.Lambda==0.0:
                raise ValueError("Lambda cannot be zero.")
            self.UniformGeneratorRange = stats.expon.cdf(numpy.array([0.0, float(self.Faces)]), scale=1.0/self.Lambda)
        elif Match.group('RotExpLambda')!=None:
            self.Distribution = self.ROTATED_EXPONENTIAL_DIST
            self._GenerateRoll = self._GenerateRollQuantile
            self._GenerateRolls = self._GenerateRollsQuantile
            self.Lambda = float(Match.group('RotExpLambda'))
            if self.Lambda==0.0:
                raise ValueError("Lambda cannot be zero.")
            self.UniformGeneratorRange = RotatedExponential.cdf(numpy.array([0.0, float(self.Faces)]), loc=float(self.Faces), scale=1.0/self.Lambda)
        else:
            self.Distribution = self.UNIFORM_DIST
            self._GenerateRoll = self._GenerateRollBasic
            self._GenerateRolls = self._GenerateRollsBasic

        self.Pdf, self.Cdf = None, None 
        #Mostly intended for potential future optimization with the analytically non-tractable normal distribution in the instance where we are interested in the number of rolls falling inside a range of values and the same object is re-used a lot to do it
        if not(LightConstructor): 
            self._GenerateRangeConditionalDistributions()
    
    def __repr__(self):
        Repr = "Generator String " + self.GeneratorString
        Repr = "\nRolls: " + str(self.Rolls)
        Repr = Repr + "\nFaces: "+ str(self.Faces)
        Repr = Repr + "\nSum: " + ("Yes" if self.Sum else "No")
        Repr = Repr + "\nDistribution: " + self._GetDistributionString()
        if self.Distribution == self.NORMAL_DIST:
            Repr = Repr + "\nUniform sample range: " + str(self.UniformGeneratorRange)
            Repr = Repr + "\nUniform sample range length: " + str(self.UniformGeneratorRange[1]-self.UniformGeneratorRange[0])
        if self.Pdf != None:
            Repr = Repr + "\nConditional Pdf over range: " + str(self.RangeConditionalDiscretePdf) + "\nConditional Cdf over range: " + str(self.Cdf) 
        return Repr
    
    def _GenerateRangeConditionalDistributions(self):
        if self.Distribution == self.NORMAL_DIST:
            CdfSource = stats.norm
            Params = {'loc': self.Mean, 'scale': self.SD}
        elif self.Distribution == self.EXPONENTIAL_DIST:
            CdfSource = stats.expon
            Params = {'scale': 1.0/self.Lambda}
        elif self.Distribution == self.ROTATED_EXPONENTIAL_DIST:
            CdfSource = RotatedExponential
            Params = {'scale':  1.0/self.Lambda, 'loc': float(self.Faces)}
        elif self.Distribution == self.UNIFORM_DIST:
            self.Pdf = numpy.repeat(1.0/self.Faces, self.Faces)
            self.Cdf = self.Pdf.cumsum()
            return
        Params['x'] = numpy.arange(1, self.Faces+1, dtype=numpy.float)
        self.Cdf = RangeConditionalCdf(Distribution=CdfSource, Min=0.0, Max=float(self.Faces), **Params)
        self.Pdf = FromCdfToPdf(self.Cdf)
            
    def _GetDistributionString(self):
        if self.Distribution == self.UNIFORM_DIST:
            return "u(0,"+str(self.Faces)+")"
        elif self.Distribution == self.NORMAL_DIST:
            return "n("+str(self.Mean)+","+str(self.SD)+")"
        elif self.Distribution == self.EXPONENTIAL_DIST:
            return "exp("+str(self.Lambda)+")"
        else:
            return "Rexp("+str(self.Lambda)+")"

    def _GenerateRollBasic(self):
        if self.Distribution == self.UNIFORM_DIST:
            return numpy.random.randint(1, self.Faces+1)

    def _GenerateRollTrialError(self):
        if self.Distribution == self.NORMAL_DIST:
            Sample = numpy.random.normal(self.Mean, self.SD)
            while Sample < 0.0 or Sample > float(self.Faces):
                Sample = numpy.random.normal(self.Mean, self.SD)
        elif self.Distribution == self.EXPONENTIAL_DIST:
            Sample = numpy.random.exponential(scale=1.0/self.Lambda)
            while Sample > float(self.Faces):
                Sample = numpy.random.exponential(scale=1.0/self.Lambda)
        else:
            return None
        return min(int(Sample)+1, self.Faces)
    
    #Was tempted to just plug self.RangeConditionalDiscretePdf in a scipy.stats.rv_discrete object and call rvs on the instance, but strangely, 
    #the following Python coded O(log(n)) search algorithm on the numpy array ran an incredulous ~30 times faster on my manual performance tests.
    def _GenerateRollCdfSearch(self):
        if self.RangeConditionalDiscreteCdf != None:
            Sample = numpy.random.uniform(0.0, 1.0)
            Bottom = 0
            Top = self.Faces-1
            Index = self.Faces//2
            while True:
                if Index == Bottom or (Sample <= self.RangeConditionalDiscreteCdf[Index] and Sample > self.RangeConditionalDiscreteCdf[Index-1]):
                    break
                elif Sample < self.RangeConditionalDiscreteCdf[Index]:
                    Top = Index
                    Index = (Index+Bottom)//2
                elif Sample > self.RangeConditionalDiscreteCdf[Index]:
                    Bottom = Index                      
                    Index = (Index+Top+1)//2
            return Index+1
    
    def _GenerateRollQuantile(self):
        if self.UniformGeneratorRange != None:
            Sample = numpy.random.uniform(self.UniformGeneratorRange[0], self.UniformGeneratorRange[1])
            if self.Distribution == self.NORMAL_DIST:
                return int(stats.norm.ppf(Sample, loc = self.Mean, scale = self.SD))+1
            elif self.Distribution == self.EXPONENTIAL_DIST:
                return int(Exponential.ppf(Sample, scale=1.0/self.Lambda))+1
            elif self.Distribution == self.ROTATED_EXPONENTIAL_DIST:
                return int(RotatedExponential.ppf(Sample, loc=float(self.Faces), scale=1.0/self.Lambda))+1
            
    def _GenerateRollsBasic(self):
        if self.Distribution == self.UNIFORM_DIST:
            return numpy.random.randint(1, self.Faces+1, self.Rolls)
                    
    def _GenerateRollsTrialError(self):
        if self.Distribution == self.NORMAL_DIST:
            Samples = numpy.random.normal(self.Mean, self.SD, size=self.Rolls)
            Samples[Samples>float(self.Faces)]=-1.0
            OutofRangeAmount = Samples[Samples<0.0].size
            while OutofRangeAmount > 0:
                Samples[Samples<0.0]  = numpy.random.normal(self.Mean, self.SD, OutofRangeAmount)
                Samples[Samples>float(self.Faces)]=-1.0
                OutofRangeAmount = Samples[Samples<0.0].size
        elif self.Distribution == self.EXPONENTIAL_DIST:
            Samples = numpy.random.exponential(scale=1.0/self.Lambda, size=self.Rolls)
            OutofRangeAmount = Samples[Samples>float(self.Faces)].size
            while OutofRangeAmount > 0:
                Samples[Samples>float(self.Faces)] = numpy.random.exponential(scale=1.0/self.Lambda, size=OutofRangeAmount)
                OutofRangeAmount = Samples[Samples>float(self.Faces)].size
        else:
            return None
        return numpy.minimum(Samples.astype(numpy.int)+1, self.Faces)
            
       
    def _GenerateRollsQuantile(self):
        if self.UniformGeneratorRange != None:
            Samples = numpy.random.uniform(self.UniformGeneratorRange[0], self.UniformGeneratorRange[1], size=self.Rolls)
            if self.Distribution == self.NORMAL_DIST:
                return stats.norm.ppf(Samples, loc = self.Mean, scale = self.SD).astype(numpy.int)+1
            elif self.Distribution == self.EXPONENTIAL_DIST:
                return Exponential.ppf(Samples, scale=1.0/self.Lambda).astype(numpy.int)+1
            elif self.Distribution == self.ROTATED_EXPONENTIAL_DIST:
                return RotatedExponential.ppf(Samples, loc=float(self.Faces), scale=1.0/self.Lambda).astype(numpy.int)+1
    
    def GenerateRolls(self):
        if self.Rolls == 1:
            return self._GenerateRoll()
        else:
            Result = self._GenerateRolls()
            if self.HighLowAmount > 0:
                Result.sort()
                if self.Descending: 
                    Result = Result[::-1]
                Result = Result[:self.HighLowAmount]
            if self.Sum:
                Result = Result.sum()
            return Result
    
    #To implement later
    def GenerateNumberRollsInRange(self, Low, High):
        pass    
    
    def _GetMomentAboutOrigin(self, Moment):
        pass
    
    def GetExpectedValue(self):
        pass
    
    def GetVariance(self):
        pass
