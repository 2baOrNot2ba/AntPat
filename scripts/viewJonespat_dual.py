#!/usr/bin/env python
"""A simple viewer for Jones patterns for dual-polarized representations.
"""
import sys
import os
import argparse
import numpy
from urlparse import urlparse
from antpat.reps.sphgridfun.tvecfun import TVecFields
from antpat.reps.sphgridfun.pntsonsphere import ZenHemisphGrid
from antpat.radfarfield import RadFarField
from antpat.dualpolelem import DualPolElem, jones2gIXR, IXRJ2IXRM
from antpat.reps import hamaker
import matplotlib.pyplot as plt 

from dreambeam.telescopes.LOFAR.telwizhelper import read_LOFAR_HAcc, convLOFARcc2HA
import antpat.io.filetypes as antfiles
from antpat.dualpolelem import DualPolElem

def plotJonesCanonical(theta, phi, jones, dpelemname):
    g, IXRJ=jones2gIXR(jones)
    IXRM=IXRJ2IXRM(IXRJ)
    IXR=IXRJ
    
    fig = plt.figure()
    fig.suptitle(dpelemname)
    plt.subplot(121,polar=False)
    plt.pcolormesh(theta, phi, 20*numpy.log10(g))
    plt.colorbar()
    plt.title('Amp gain')
    plt.subplot(122,polar=False)
    plt.pcolormesh(theta, phi, 10*numpy.log10(IXR))
    plt.colorbar()
    plt.title('IXR_J')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dualpol_file",
                 help='Filename of dual-polarization FF. (Hamaker-Arts or two polarization requests)')
    parser.add_argument("freq", nargs='?', type=float,
                        help="Frequency in Hertz")
    args = parser.parse_args()
    
    if args.dualpol_file.endswith(antfiles.HamArtsuffix):
        artsdata = read_LOFAR_HAcc(args.dualpol_file)
        artsdata['channels'] = [args.freq]
        hp = hamaker.HamakerPolarimeter(artsdata)
    elif args.dualpol_file.endswith(antfiles.FEKOsuffix):
        hp = DualPolElem()
        hp.load_ffe(args.dualpol_file)
    else:
        print("dual-pol pattern file type not known")
        exit(1)
    THETA, PHI = ZenHemisphGrid()
    jones=hp.getJonesAlong([args.freq], (THETA, PHI) )
    plotJonesCanonical(THETA, PHI, jones, os.path.basename(args.dualpol_file)
                       +' ('+str(args.freq/1e6)+' MHz)')