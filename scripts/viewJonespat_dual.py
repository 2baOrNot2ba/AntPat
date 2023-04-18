#!/usr/bin/env python3
"""A simple viewer for Jones patterns for dual-polarized representations.
"""
import sys
import argparse
import numpy
import matplotlib.pyplot as plt

from antpat.reps.sphgridfun.pntsonsphere import ZenHemisphGrid
from antpat.dualpolelem import DualPolElem, jones2gIXR, IXRJ2IXRM
from antpat.io.dualpol_ingest import load_dualpol_files


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
        # nrlvls = 5
        # g_lvls = numpy.max(g) - 3.0*numpy.arange(nrlvls)
    plt.pcolormesh(phi, numpy.rad2deg(theta), g)
    # plt.contour( phi, numpy.rad2deg(theta), g_dress, levels = g_lvls)
    plt.colorbar()
    plt.title('Amp gain')
    plt.subplot(122, polar=polarplt)
    plt.pcolormesh(phi, numpy.rad2deg(theta), 10*numpy.log10(IXR))
    plt.colorbar()
    plt.title('IXR_J')
    plt.show()


def plot_jonespat(thetas, phis, jones):
    from antpat.reps.sphgridfun import tvecfun
    for polchan in [0, 1]:
        E_th = jones[:, :, polchan, 0].squeeze()
        E_ph = jones[:, :, polchan, 1].squeeze()
        tvecfun.plotvfonsph(thetas, phis, E_th, E_ph, args.freq,
                            vcoordlist=['Ludwig3'], projection='orthographic')

def print_jonespat(freq, theta, phi, jones):
    print(','.join([str(f) for f in jones.flatten()]))#.tolist()))


if __name__ == "__main__":
    cmd = sys.argv.pop(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--theta", type=float, default=None)
    parser.add_argument("-p","--phi", type=float, default=None)
    parser.add_argument("freq", type=float,
                        help="Frequency in Hertz")
    parser.add_argument("filename", help="""
        Filename of dual-polarization FF, Hamaker-Arts format,
        or a single-polarization FF (p-channel)""")
    parser.add_argument("filename_q", nargs='?',
                        help="""
                        Filename of second (q-channel) single-polarization FF.
                        """)
    args = parser.parse_args()

    dpe, _, _ = load_dualpol_files(args.filename, args.filename_q)
    if args.theta is None and args.phi is None:
        thetas, phis = ZenHemisphGrid()
    else:
        thetas, phis = [args.theta], [args.phi]

    jones = dpe.getJonesAlong([args.freq], (thetas, phis))

    if cmd == 'plot':
        plot_jonespat(thetas, phis, jones)
    else:
        print_jonespat(args.freq, thetas[0], phis[0], jones)

