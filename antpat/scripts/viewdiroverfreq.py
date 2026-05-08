#!/usr/bin/python3
"""View far-field for a given direction over frequency."""
import argparse
import numpy
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from antpat.reps.sphgridfun import tvecfun
from antpat.radfarfield import RadFarField
from antpat.reps.vsharm.vshfield import vshField
from antpat.reps.vsharm.coefs import load_SWE2vshCoef
import antpat.io.filetypes as antfiles
from antpat.io.NECread import readNECout_tvecfuns


parser = argparse.ArgumentParser()
parser.add_argument("patternURL",
                    help="""Path to pattern file.
(Use format: filepath[#request])""")
parser.add_argument("theta", type=float,
                    help="theta coordinate of direction [radians]")
parser.add_argument("phi", type=float,
                    help="phi coordinate of direction [radians]")
args = parser.parse_args()
pattern_URL = urlparse(args.patternURL)
FFfile = pattern_URL.path
request = pattern_URL.fragment
if request == '':
    request = None
theta = numpy.array([args.theta])
phi = numpy.array([args.phi])
if FFfile.endswith(antfiles.FEKOsuffix):
    tvf = tvecfun.TVecFields()
    tvf.load_ffe(FFfile, request)
elif FFfile.endswith(antfiles.GRASPsuffix):
    cfs, freqs = load_SWE2vshCoef(FFfile, convention='FEKO')
    antff = RadFarField(vshField([cfs], freqs))
elif FFfile.endswith(antfiles.NECsuffix):
    tvf = readNECout_tvecfuns(FFfile)
else:
    raise RuntimeError("Far-field pattern file type not known")
antff = RadFarField(tvf)
freqs = antff.getfreqs()
E_th, E_ph = antff.getFFalong(freqs, (theta, phi))
# plt.plot(freqs, abs(E_th), 'b-')
# plt.plot(freqs, abs(E_ph), 'r-')
plt.plot(freqs, abs(E_th)**2+abs(E_ph)**2, 'r-')
plt.show()
