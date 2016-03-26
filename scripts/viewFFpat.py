#!/usr/bin/python
import sys
from antpat.reps.sphgridfun import tvecfun

FEKOsuffix = 'ffe'
GRASPsuffix = 'swe'
NECsuffix = 'out'

def plotFEKO(filename, request=None):
    tvf = tvecfun.TVecFields()
    tvf.load_ffe(filename, request)
    freqs = tvf.getRs()
    freq = freqs[0]
    (THETA, PHI, E_th, E_ph) = (tvf.getthetas(), tvf.getphis(), tvf.getFthetas(freq), tvf.getFphis(freq))
    tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph, freq, vcoord='Ludwig3', projection='orthographic')

if __name__ == "__main__":
    FFfile = sys.argv[1]
    if FFfile.endswith(FEKOsuffix):
        if len(sys.argv) > 2:
            request = sys.argv[2]
        else:
            request = None
        plotFEKO(FFfile, request)
    elif FFfile.endswith(GRASPsuffix):
        print("Not implemented yet.")
    elif FFfile.endswith(NECsuffix):
        print("Not implemented yet.")
    else:
        print("Far-field pattern file type not known")
        exit(1)
