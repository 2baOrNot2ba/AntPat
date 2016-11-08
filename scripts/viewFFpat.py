#!/usr/bin/python
"""A simple viewer for legacy far-field pattern files."""
import sys
import argparse
import math
import numpy
from urlparse import urlparse
from antpat.reps.sphgridfun import tvecfun
from antpat.radfarfield import RadFarField
from antpat.reps.vsharm.vshfield import vshField
from antpat.reps.vsharm.coefs import load_SWE2vshCoef


FEKOsuffix = 'ffe'
GRASPsuffix = 'sph'
NECsuffix = 'out'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patternURL",
                 help='Path to pattern file. Use format: filepath[#request]')
    parser.add_argument("freq", nargs='?', type=float,
                        help="Frequency in Hertz")
    args = parser.parse_args()
    pattern_URL = urlparse(args.patternURL)
    FFfile = pattern_URL.path
    request = pattern_URL.fragment
    if request == '': request = None
    freq = args.freq
    if FFfile.endswith(FEKOsuffix):
        tvecfun.plotFEKO(FFfile, request, freq)
    elif FFfile.endswith(GRASPsuffix):
        cfs, freq = load_SWE2vshCoef(FFfile, convention='FEKO')
        antFF = RadFarField(vshField([cfs], [freq]))
        THETA, PHI, V_th, V_ph = antFF.getFFongrid(freq)
        c = 3.0e8
        k = (2*math.pi*freq)/c
        Z0 = 376.7
        V2EfieldNrm = k*numpy.sqrt(Z0/(2*2*math.pi)) #Not sure about a 1/sqrt(2) factor
        E_th = V2EfieldNrm*V_th
        E_ph = V2EfieldNrm*V_ph
        #plotAntPat2D(theta_var,phi_fix,E_th,E_ph)
        tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph, freq,
                            vcoord='sph',projection='equirectangular')
    elif FFfile.endswith(NECsuffix):
        print("Not implemented yet.")
    else:
        print("Far-field pattern file type not known")
        exit(1)
