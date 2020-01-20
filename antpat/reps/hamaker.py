#!/usr/bin/python
"""Hamaker's analytic antenna pattern model."""
# TobiaC 2015-11-29 (2015-07-31)

import sys
import math
import re
import pickle
import numpy
from antpat import dualpolelem
from antpat.reps.sphgridfun import tvecfun, pntsonsphere
import matplotlib.pyplot as plt
HA_LBAfile_default = ''
PICKLE_PROTO = pickle.HIGHEST_PROTOCOL


class HamakerPolarimeter(object):
    """This is the Hamaker polarimeter model class."""

    nr_pols = 2  # Number of polarization channels

    def __init__(self, artsdata):
        """Objects are created based on a Arts coefficient C++ header
        file. There is current one default set for the HBA and one for
        LBA."""
        self.coefs = artsdata['coefs']
        self.HAcoefversion = artsdata['HAcoefversion']
        self.HAcoefband = artsdata['HAcoefband']
        self.HAcoefnrelem = artsdata['HAcoefnrelem']
        self.freq_center = artsdata['freq_center']
        self.freq_range = artsdata['freq_range']
        self.channels = artsdata['channels']

        self.nr_bands = len(self.coefs)
        self.freqintervs = (self.freq_center-self.freq_range,
                            self.freq_center+self.freq_range)

    def getfreqs(self):
        """Returns nominals channel center frequencies"""
        return self.channels

    def getJonesAlong(self, freqvals, theta_phi):
        """Compute Jones matrix for given frequencies and directions.
        Input is list of frequencies in Hz and a list of theta,phi pairs;
        and the output is Jones[freq, dir_th, dir_ph, polchan, comp]."""
        mask_horizon = True
        (theta, phi) = theta_phi
        theta = numpy.array(theta)
        phi = numpy.array(phi)
        freqvals = numpy.array(freqvals)
        (k_ord, TH_ord, FR_ord, nr_pol) = self.coefs.shape
        freqn = (freqvals-self.freq_center)/self.freq_range
        if len(freqvals) > 1:
            frqXdrn_shp = freqvals.shape+theta.shape
        else:
            frqXdrn_shp = theta.shape
        response = numpy.zeros(frqXdrn_shp+(2, 2), dtype=complex)
        for ki in range(k_ord):
            P = numpy.zeros((nr_pol,)+frqXdrn_shp, dtype=complex)
            for THi in range(TH_ord):
                for FRi in range(FR_ord):
                    fac = numpy.multiply.outer(freqn**FRi,
                                               theta**THi).squeeze()
                    P[0, ...] += self.coefs[ki, THi, FRi, 0]*fac
                    P[1, ...] += self.coefs[ki, THi, FRi, 1]*fac
            ang = (-1)**ki*(2*ki+1)*phi
            response[..., 0, 0] += +numpy.cos(ang)*P[0, ...]
            response[..., 0, 1] += -numpy.sin(ang)*P[1, ...]
            response[..., 1, 0] += +numpy.sin(ang)*P[0, ...]
            response[..., 1, 1] += +numpy.cos(ang)*P[1, ...]
            # numpy.array([[math.cos(ang)*P[0],-math.sin(ang)*P[1]],
            #              [math.sin(ang)*P[0], math.cos(ang)*P[1]]])
        # Mask beam below horizon
        if mask_horizon:
            mh = numpy.ones(frqXdrn_shp+(1, 1))
            mh[..., numpy.where(theta > numpy.pi/2), 0, 0] = 0.0
            response = mh*response
        return response


def read_LOFAR_HAcc(coefsccfilename):
    """Read Hamaker-Arts coefficients from c++ header files used in the
    "lofar_element_response" code developed at ASTRON for LOFAR.

    These header files contains LOFAR specific constructs such as reference
    to "lba" and "hba", so it is not suitable for other projects.
    """
    NR_POLS = 2
    re_fcenter = r'[lh]ba_freq_center\s*=\s*(?P<centerstr>.*);'
    re_frange = r'[lh]ba_freq_range\s*=\s*(?P<rangestr>.*);'
    re_shape = \
        r'default_[lh]ba_coeff_shape\[3\]\s*=\s*\{(?P<lstshp>[^\}]*)\}'
    re_hl_ba_coeffs_lst = \
        r'(?P<version>\w+)(?P<band>[hl]ba)_coeff\s*\[\s*(?P<nrelem>\d+)\s*\]\s*=\s*\{(?P<cmplstr>[^\}]*)\}'
    re_cc_cmpl_coef = r'std::complex<double>\((.*?)\)'
    with open(coefsccfilename, 'r') as coefsccfile:
        coefsfile_content = coefsccfile.read()
    searchres = re.search(re_fcenter, coefsfile_content)
    freq_center = float(searchres.group('centerstr'))
    searchres = re.search(re_frange, coefsfile_content)
    freq_range = float(searchres.group('rangestr'))
    searchres = re.search(re_shape, coefsfile_content)
    lstshp = [int(lstshpel) for lstshpel in
              searchres.group('lstshp').split(',')]
    lstshp.append(NR_POLS)
    searchres = re.search(re_hl_ba_coeffs_lst, coefsfile_content, re.M)
    HAcoefversion = searchres.group('version')
    HAcoefband = searchres.group('band')
    HAcoefnrelem = searchres.group('nrelem')
    lstofCmpl = re.findall(re_cc_cmpl_coef, searchres.group('cmplstr'))
    cmplx_lst = []
    for reimstr in lstofCmpl:
        reimstrs = reimstr.split(',')
        cmplx_lst.append(complex(float(reimstrs[0]), float(reimstrs[1])))
    coefs = numpy.reshape(numpy.array(cmplx_lst), lstshp)
    # The coefficients are order now as follows:
    #   coefs[k,theta,freq,spherical-component].shape == (2,5,5,2)
    artsdata = {'coefs': coefs, 'HAcoefversion': HAcoefversion,
                'HAcoefband': HAcoefband, 'HAcoefnrelem': HAcoefnrelem,
                'freq_center': freq_center, 'freq_range': freq_range}
    return artsdata


def convLOFARcc2HA(inpfile, outfile, channels):
    """Convert a .cc file of the Hamaker-Arts model to a file with a pickled
    dict of a Hamaker-Arts instance."""
    artsdata = read_LOFAR_HAcc(inpfile)
    artsdata['channels'] = channels
    pickle.dump(artsdata, open(outfile, 'wb'), PICKLE_PROTO)


def convHA2DPE(inp_HA_file, out_DP_file):
    """Convert a file with a pickled dict of a Hamaker-Arts instance to a
    file with a pickled DualPolElem object."""
    artsdata = pickle.load(open(inp_HA_file, 'rb'))
    HLBA = HamakerPolarimeter(artsdata)
    stnDPolel = dualpolelem.DualPolElem(HLBA)
    pickle.dump(stnDPolel, open(out_DP_file, 'wb'), PICKLE_PROTO)


def plotElemPat(artsdata, frequency=55.0e6):
    """Plots the HA antenna pattern over the entire Hemisphere."""
    THETA, PHI = pntsonsphere.ZenHemisphGrid()  # theta=0.2rad for zenith anomaly
    hp = HamakerPolarimeter(artsdata)
    jones = hp.getJonesAlong([frequency], (THETA, PHI))
    EsTh = numpy.squeeze(jones[..., 0, 0])
    EsPh = numpy.squeeze(jones[..., 0, 1])
    tvecfun.plotvfonsph(THETA, PHI, EsTh, EsPh, freq=frequency,
                        vcoordlist=['sph'], projection='azimuthal-equidistant',
                        vfname='Hamaker')
    EsTh = numpy.squeeze(jones[..., 1, 0])
    EsPh = numpy.squeeze(jones[..., 1, 1])
    tvecfun.plotvfonsph(THETA, PHI, EsTh, EsPh, freq=frequency,
                        vcoordlist=['sph'], projection='equirectangular',
                        vfname='Hamaker')  # vcoordlist=['Ludwig3']


def showAnomaly():
    """Demostrates the anomaly of the Hamaker-Arts model close to zenith."""
    frequency = 225e6
    nrPnts = 200
    timeAng = 0.5
    timeAngs = numpy.linspace(-timeAng, timeAng, nrPnts)/2.0
    theta0 = 0.5
    thetas, phis = pntsonsphere.getTrack(theta0, 0*math.pi/4, theta0-0.001,
                                         timeAngs)
    hp = HamakerPolarimeter(artsdata)
    jones = hp.getJonesAlong([frequency], (phis+1*5*math.pi/4, thetas))
    EsTh = numpy.squeeze(jones[..., 0, 0])
    EsPh = numpy.squeeze(jones[..., 0, 1])
    plt.subplot(2, 1, 1)
    plt.plot(phis/math.pi*180, 90-thetas/math.pi*180, '*')
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    plt.subplot(2, 1, 2)
    plt.plot(timeAngs*60, numpy.abs(EsTh))
    plt.xlabel('Transit time [min]')
    plt.ylabel('Gain [rel.]')
    plt.show()


def getJones(freq, az, el):
    """Print the Jones matrix of the HA model for a frequency and direction."""
    hp = HamakerPolarimeter(HA_LBAfile_default)
    jones = hp.getJonesAlong([freq], (0.1, 0.2))
    print("Jones:")
    print(jones)
    print("J.J^H:")
    print(numpy.dot(jones, jones.conj().transpose()).real)
    IXRJ = dualpolelem.getIXRJ(jones)
    print("IXRJ:", 10*numpy.log10(IXRJ), "[dB]")


def _getargs():
    freq = float(sys.argv[1])
    az = float(sys.argv[2])
    el = float(sys.argv[3])
    return freq, az, el


if __name__ == "__main__":
    from dreambeam.telescopes.LOFAR.telwizhelper import read_LOFAR_HAcc
    artsdata = read_LOFAR_HAcc('../../example_FF_files/DefaultCoeffLBA.cc')
    freq = 80e6
    artsdata['channels'] = [freq]
    plotElemPat(artsdata, freq)
    # showAnomaly()
    # HBAmod = HamakerPolarimeter(HA_HBAfile_default)
    # jones = HBAmod.getJonesAlong([150e6, 160e6, 170e6], ( [0.1,0.1], [0.3, 0.4]) )
    # print(jones)
