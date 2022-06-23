#!/usr/bin/python
"""A simple viewer for legacy far-field pattern files."""
import argparse
import math
import numpy
import os.path
from urllib.parse import urlparse
from antpat.reps.sphgridfun import tvecfun
from antpat.radfarfield import RadFarField
from antpat.reps.vsharm.vshfield import vshField
from antpat.reps.vsharm.coefs import load_SWE2vshCoef
import antpat.io.filetypes as ft
from antpat.io.NECread import readNECout_tvecfuns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patternURL",
                        help="""Path to pattern file.
(Use format: filepath[#request])""")
    parser.add_argument("freq", nargs='?', type=float,
                        help="Frequency in Hertz")
    args = parser.parse_args()
    pattern_URL = urlparse(args.patternURL)
    FFfile = pattern_URL.path
    request = pattern_URL.fragment
    if request == '':
        request = None
    freq = args.freq
    if FFfile.endswith(ft.FEKOsuffix):
        from antpat.reps.sphgridfun.tvecfun import TVecFields, plotvfonsph
        tvf = TVecFields()
        tvf.load_ffe(FFfile, request)
        freqs = tvf.getRs()
        if freq is None:
            frqIdx = 0
        else:
            frqIdx = int(numpy.interp(freq, freqs, range(len(freqs))))
        freq = freqs[frqIdx]
        print("Frequency={}".format(freq))
        (THETA, PHI, E_th, E_ph) = (tvf.getthetas(), tvf.getphis(),
                                    tvf.getFthetas(freq), tvf.getFphis(freq))
        tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph, freq=freq,
                            vcoordlist=['Ludwig3', 'circ'],
                            projection='azimuthal-equidistant',
                            cmplx_rep='AbsAng',
                            vfname=os.path.basename(FFfile))
    elif FFfile.endswith(ft.GRASPsuffix):
        cfs, freq = load_SWE2vshCoef(FFfile, convention='FEKO')
        antFF = RadFarField(vshField([cfs], [freq]))
        THETA, PHI, V_th, V_ph = antFF.getFFongrid(freq)
        c = 3.0e8
        k = (2*math.pi*freq)/c
        Z0 = 376.7
        V2EfieldNrm = k*numpy.sqrt(Z0/(2*2*math.pi)) #Not sure about a 1/sqrt(2) factor
        E_th = V2EfieldNrm*V_th
        E_ph = V2EfieldNrm*V_ph
        tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph, freq,
                            vcoord='sph', projection='equirectangular')
    elif FFfile.endswith(ft.NECsuffix):
        tvf = readNECout_tvecfuns(FFfile)
        (THETA, PHI, E_th, E_ph) = (tvf.getthetas(), tvf.getphis(),
                                    tvf.getFthetas(freq), tvf.getFphis(freq))
        tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph, freq=freq,
                            vcoordlist=['Ludwig3'],
                            projection='orthographic',
                            cmplx_rep='AbsAng',
                            vfname=os.path.basename(FFfile))
    else:
        print("Far-field pattern file type not known")
        exit(1)
