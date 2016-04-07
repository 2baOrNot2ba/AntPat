#!/usr/bin/python
"""A simple viewer for legacy far-field pattern files."""
import sys
from antpat.reps.sphgridfun import tvecfun


FEKOsuffix = 'ffe'
GRASPsuffix = 'swe'
NECsuffix = 'out'


if __name__ == "__main__":
    FFfile = sys.argv[1]
    if FFfile.endswith(FEKOsuffix):
        if len(sys.argv) > 2:
            request = sys.argv[2]
        else:
            request = None
        tvecfun.plotFEKO(FFfile, request)
    elif FFfile.endswith(GRASPsuffix):
        print("Not implemented yet.")
    elif FFfile.endswith(NECsuffix):
        print("Not implemented yet.")
    else:
        print("Far-field pattern file type not known")
        exit(1)
