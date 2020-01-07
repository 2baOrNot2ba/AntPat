"""Objects of this class are to be imported as Radiation Far-fields."""
import numpy


class vshField(object):
    def __init__(self, vshcoefs, frequencies=[0.0]):
        self.frequencies = frequencies
        self.vshcoefs = vshcoefs

    def getCoefAt(self, freq):
        return self.vshcoefs[self.getfreqidx(freq)]

    def getfreqidx(self, freqval):
        frequencies = numpy.array(self.frequencies)
        freqidxlst = numpy.where(frequencies == freqval)
        if freqidxlst == []:
            raise RuntimeError("Frequency not found")
        freqidx = freqidxlst[0][0]  # For now assume unique value.
        return freqidx
