#!/usr/bin/env python
"""A simple viewer for Jones patterns based on two far-field pattern files.
(Possibly based on one FF pattern files if it has two requests: one for each
polarization channel.)"""
import sys
import os
import argparse
from urlparse import urlparse
from antpat.reps.sphgridfun.tvecfun import TVecFields
from antpat.radfarfield import RadFarField
from antpat.dualpolelem import DualPolElem


FEKOsuffix = 'ffe'
GRASPsuffix = 'swe'
NECsuffix = 'out'

def plotJones_fromFEKOfiles(p_chan_file, q_chan_file, freq):
    (tvf_p, tvf_q) = (TVecFields(), TVecFields())
    tvf_p.load_ffe(p_chan_file)
    tvf_q.load_ffe(q_chan_file)
    (ant_p, ant_q) = (RadFarField(tvf_p), RadFarField(tvf_q))
    (p_chan_name, q_chan_name) = (os.path.basename(p_chan_file), os.path.basename(q_chan_file))
    (ant_p.name, ant_q.name) = (p_chan_name, q_chan_name)
    dualpolAnt = DualPolElem(ant_p, ant_q)
    dualpolAnt.plotJonesPat3D(freq, vcoord='circ',
                              projection='azimuthal-equidistant',
                              cmplx_rep='ReIm')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("p_chan_file",
                 help='Filename of polarization channel p')
    parser.add_argument("q_chan_file",
                 help='Filename of polarization channel p')
    parser.add_argument("freq", nargs='?', type=float,
                        help="Frequency in Hertz")
    args = parser.parse_args()
    
    if args.p_chan_file.endswith(FEKOsuffix):
        plotJones_fromFEKOfiles(args.p_chan_file, args.q_chan_file, args.freq)
    elif args.p_chan_file.endswith(GRASPsuffix):
        print("Not implemented yet.")
    elif args.p_chan_file.endswith(NECsuffix):
        print("Not implemented yet.")
    else:
        print("Far-field pattern file type not known")
        exit(1)
