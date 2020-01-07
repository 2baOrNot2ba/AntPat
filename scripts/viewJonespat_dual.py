#!/usr/bin/env python
"""A simple viewer for Jones patterns for dual-polarized representations.
"""
import os
import argparse
import numpy
import matplotlib.pyplot as plt
from urlparse import urlparse

from antpat.reps.sphgridfun.tvecfun import TVecFields
from antpat.reps.sphgridfun.pntsonsphere import ZenHemisphGrid
from antpat.radfarfield import RadFarField
from antpat.dualpolelem import DualPolElem, jones2gIXR, IXRJ2IXRM
from antpat.reps import hamaker
from dreambeam.telescopes.LOFAR.telwizhelper import read_LOFAR_HAcc
import antpat.io.filetypes as antfiles


def plotJonesCanonical(theta, phi, jones, dpelemname):
    normalize = True
    dbscale = True
    polarplt = True
    IXRTYPE = 'IXR_J'  # Can be IXR_J or IXR_M

    g, IXRJ = jones2gIXR(jones)
    IXRM = IXRJ2IXRM(IXRJ)
    if IXRTYPE == 'IXR_J':
        IXR = IXRJ
    elif IXRTYPE == 'IXR_J':
        IXR = IXRM
    else:
        raise RuntimeError("""Error: IXR type {} unknown.
                           Known types are IXR_J, IXR_M.""".format(IXRTYPE))

    fig = plt.figure()
    fig.suptitle(dpelemname)
    plt.subplot(121, polar=polarplt)
    if normalize:
        g_max = numpy.max(g)
        g = g/g_max
    if dbscale:
        g = 20*numpy.log10(g)
        nrlvls = 5
        g_lvls = numpy.max(g) - 3.0*numpy.arange(nrlvls)
    plt.pcolormesh(phi, numpy.rad2deg(theta), g)
    # plt.contour( phi, numpy.rad2deg(theta), g_dress, levels = g_lvls)
    plt.colorbar()
    plt.title('Amp gain')
    plt.subplot(122, polar=polarplt)
    plt.pcolormesh(phi, numpy.rad2deg(theta), 10*numpy.log10(IXR))
    plt.colorbar()
    plt.title('IXR_J')
    plt.show()


def plotFFpat():
    E_th = jones[:, :, 0, 0].squeeze()
    E_ph = jones[:, :, 0, 1].squeeze()
    from antpat.reps.sphgridfun import tvecfun
    tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph, args.freq,
                        vcoordlist=['Ludwig3'], projection='orthographic')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("freq", type=float,
                        help="Frequency in Hertz")
    parser.add_argument("filename",
                 help = """
                 Filename of dual-polarization FF, Hamaker-Arts format,
                 or a single-polarization FF (p-channel)""")
    parser.add_argument("filename_q", nargs='?',
                 help='Filename of second (q-channel) single-polarization FF.')
    args = parser.parse_args()

    if args.filename.endswith(antfiles.HamArtsuffix):
        artsdata = read_LOFAR_HAcc(args.filename)
        artsdata['channels'] = [args.freq]
        hp = hamaker.HamakerPolarimeter(artsdata)
    elif args.filename.endswith(antfiles.FEKOsuffix):
        hp = DualPolElem()
        hp.load_ffes(args.filename, args.filename_q)
    else:
        raise RuntimeError("dual-pol pattern file type not known")
    THETA, PHI = ZenHemisphGrid()
    jones = hp.getJonesAlong([args.freq], (THETA, PHI))
    plotFFpat()
    exit()
    plotJonesCanonical(THETA, PHI, jones, os.path.basename(args.filename)
                       + ' (' + str(args.freq/1e6) + ' MHz)')
