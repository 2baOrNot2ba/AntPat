#!/usr/bin/python
"""A simple viewer for legacy far-field pattern files."""
import sys
import argparse
from urlparse import urlparse
from antpat.reps.sphgridfun import tvecfun


FEKOsuffix = 'ffe'
GRASPsuffix = 'swe'
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
        print("Not implemented yet.")
    elif FFfile.endswith(NECsuffix):
        print("Not implemented yet.")
    else:
        print("Far-field pattern file type not known")
        exit(1)
