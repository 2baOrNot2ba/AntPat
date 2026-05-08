#!/usr/bin/env python
"""A simple viewer for Stokes patterns based on two far-field pattern files.
(Possibly based on one FF pattern files if it has two requests: one for each
polarization channel.)"""
import os
import argparse
import numpy
import matplotlib.pyplot as plt
from antpat.reps.sphgridfun.tvecfun import TVecFields
from antpat.radfarfield import RadFarField
from antpat.dualpolelem import DualPolElem

FEKOsuffix = 'ffe'
GRASPsuffix = 'swe'
NECsuffix = 'out'


def Jones2Stokes(Jones):
    """Convert Jones matrix to Stokes vector. This assumes dual-pol antenna receiving unpolarized unit
    valued radiation i.e. incoming Stokes = (1,0,0,0)."""
    brightmat = numpy.matmul(Jones, numpy.swapaxes(numpy.conjugate(Jones),-1,-2))
    StokesI = numpy.real(brightmat[...,0,0]+brightmat[...,1,1])
    StokesQ = numpy.real(brightmat[...,0,0]-brightmat[...,1,1])
    StokesU = numpy.real(brightmat[...,0,1]+brightmat[...,1,0])
    StokesV = numpy.imag(brightmat[...,0,1]-brightmat[...,1,0])
    return StokesI, StokesQ, StokesU, StokesV


def plotStokes_fromFEKOfiles(p_chan_file, q_chan_file, freq):
    (tvf_p, tvf_q) = (TVecFields(), TVecFields())
    tvf_p.load_ffe(p_chan_file)
    tvf_q.load_ffe(q_chan_file)
    (ant_p, ant_q) = (RadFarField(tvf_p), RadFarField(tvf_q))
    (p_chan_name, q_chan_name) = (os.path.basename(p_chan_file), os.path.basename(q_chan_file))
    (ant_p.name, ant_q.name) = (p_chan_name, q_chan_name)
    dualpolAnt = DualPolElem(ant_p, ant_q)
    THETA, PHI, Jones = dualpolAnt.getJonesPat(freq)
    (StokesI, StokesQ, StokesU, StokesV) = Jones2Stokes(Jones)

    x = THETA*numpy.cos(PHI)
    y = THETA*numpy.sin(PHI)
    #x= THETA
    #y=PHI
    xyNames = ('theta*cos(phi)','theta*sin(phi)')
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    plt.pcolormesh(x, y, 10*numpy.log10(StokesI), label="I")
    #plt.pcolormesh(x, y, StokesI, label="I")
    plt.colorbar()
    ax1.set_title('I (dB)')

    ax2 = fig.add_subplot(222)
    plt.pcolormesh(x, y, StokesQ/StokesI, label="Q")
    plt.colorbar()
    ax2.set_title('Q/I')

    ax3 = fig.add_subplot(223)
    plt.pcolormesh(x, y, StokesU/StokesI, label="U")
    plt.colorbar()
    ax3.set_title('U/I')

    ax4 = fig.add_subplot(224)
    plt.pcolormesh(x, y, StokesV/StokesI, label="V")
    plt.colorbar()
    ax4.set_title('V/I')
    fig.suptitle('Stokes (azimuthal-equidistant proj) @ ' +str(freq/1e9)+' GHz')
    plt.show()


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
        plotStokes_fromFEKOfiles(args.p_chan_file, args.q_chan_file, args.freq)
    elif args.p_chan_file.endswith(GRASPsuffix):
        print("Not implemented yet.")
    elif args.p_chan_file.endswith(NECsuffix):
        print("Not implemented yet.")
    else:
        print("Far-field pattern file type not known")
        exit(1)
