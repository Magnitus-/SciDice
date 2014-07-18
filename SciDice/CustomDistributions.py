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

def RangeConditionalCdf(Distribution, Min=None, Max=None, **NameArgs):
    x = NameArgs['x']
    NameArgs['x'] = Max
    CdfMax = (1.0 if Max == None else Distribution.cdf(**NameArgs))
    NameArgs['x'] = Min
    CdfMin = (0.0 if Min == None else Distribution.cdf(**NameArgs))
    NameArgs['x'] = x
    Result = (Distribution.cdf(**NameArgs)- CdfMin)/(CdfMax-CdfMin)
    if numpy.isscalar(Result):
        Result = (Result if Result>=0.0 and Result<=1.0 else 0.0)
    else:
        Result[Result<0.0] = 0.0
        Result[Result > 1.0] = 0.0
    return Result
    
def FromCdfToPdf(CdfArray):
    CdfMinus1 = numpy.repeat(0.0, CdfArray.size)
    CdfMinus1[1:] = CdfArray[:-1]
    return CdfArray-CdfMinus1

class RotatedExponential(object):
	@staticmethod
	def cdf(x, loc, scale):
		Lambda = (1.0/scale)
		return numpy.exp(Lambda*x-Lambda*loc)
	
	@staticmethod
	def ppf(x, loc, scale):
		Lambda = (1.0/scale)
		return loc+(numpy.log(x)/Lambda)
		
class Exponential(object):
	#Wanted ot use stats.expon.ppf, but it's amazingly slow on my machine compared to this hand-written solution....	
	@staticmethod
	def ppf(q, scale):
		Lambda = (1.0/scale)
		return -numpy.log(1-q)/Lambda
