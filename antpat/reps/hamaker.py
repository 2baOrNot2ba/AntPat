#!/usr/bin/python
"""Hamaker's analytic antenna pattern model."""
#TobiaC 2015-11-29 (2015-07-31)

import sys
#sys.path.append('/home/tobia/projects/BeamFormica/AntPatter/')
import math
import cmath
import scipy.special
import numpy
from antpat import dualpolelem
from antpat.reps.sphgridfun import tvecfun, pntsonsphere
import matplotlib.pyplot as plt 


class HamakerPolarimeter(object):
    """This is the Hamaker polarimeter model class.""" 
    
    nr_pols = 2 #Number of polarization channels
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
        
        self.nr_bands=len(self.coefs)
        self.freqintervs = (self.freq_center-self.freq_range,
            self.freq_center+self.freq_range)
    
    def getfreqs(self):
        """Returns nominals channel center frequencies"""
        return self.channels
    
    def getJonesAlong(self, freqvals, theta_phi):
        """Compute Jones matrix for given frequencies and directions.
        Input is list of frequencies in Hz and a list of theta,phi pairs;
        and the output is Jones[polchan, comp, freq, direction]."""
        (theta, phi) = theta_phi
        theta = numpy.array(theta)
        phi = numpy.array(phi)
        freqvals = numpy.array(freqvals)
        (k_ord, TH_ord, FR_ord, nr_pol) = self.coefs.shape
        freqn = (freqvals-self.freq_center)/self.freq_range
        if len(freqvals) > 1:
            frqXdrn_shp = freqvals.shape+theta.shape
        else :
            frqXdrn_shp = theta.shape
        response = numpy.zeros(frqXdrn_shp+(2, 2), dtype=complex)
        for ki in range(k_ord):
            P = numpy.zeros((nr_pol,)+frqXdrn_shp, dtype=complex)
            for THi in range(TH_ord):
                for FRi in range(FR_ord):
                    fac = numpy.multiply.outer(freqn**FRi, theta**THi).squeeze()
                    P[0,...] += self.coefs[ki,THi,FRi,0]*fac
                    P[1,...] += self.coefs[ki,THi,FRi,1]*fac
            ang = (-1)**ki*(2*ki+1)*phi
            response[...,0,0] += +numpy.cos(ang)*P[0,...]
            response[...,0,1] += -numpy.sin(ang)*P[1,...]
            response[...,1,0] += +numpy.sin(ang)*P[0,...]
            response[...,1,1] += +numpy.cos(ang)*P[1,...]
            #numpy.array([[math.cos(ang)*P[0],-math.sin(ang)*P[1]],
            #             [math.sin(ang)*P[0], math.cos(ang)*P[1]]])
        return response


def plotElemPat(frequency = 55.0e6):
    """Plots the HA antenna pattern over the entire Hemisphere."""
    THETA, PHI = pntsonsphere.ZenHemisphGrid() #theta=0.2rad for zenith anomaly
    hp = HamakerPolarimeter(HA_LBAfile_default)
    jones=hp.getJonesAlong([frequency], (THETA, PHI) )
    EsTh = numpy.squeeze(jones[...,0,0])
    EsPh = numpy.squeeze(jones[...,0,1])
    tvecfun.plotvfonsph(THETA, PHI, EsTh, EsPh, freq=frequency, vcoord='Ludwig3')
    EsTh = numpy.squeeze(jones[...,1,0])
    EsPh = numpy.squeeze(jones[...,1,1])
    tvecfun.plotvfonsph(THETA, PHI, EsTh, EsPh, freq=frequency, vcoord='Ludwig3')


def showAnomaly():
    """Demostrates the anomaly of the Hamaker-Arts model close to zenith."""
    frequency = 225e6
    nrPnts = 200
    timeAng = 0.5
    timeAngs = numpy.linspace(-timeAng, timeAng, nrPnts)/2.0
    theta0 = 0.5
    thetas, phis = pntsonsphere.getTrack(theta0, 0*math.pi/4, theta0-0.001, timeAngs)
    hp = HamakerPolarimeter(HA_LBAfile_default)
    #jones = hp.getJonesAlong([frequency], (phis+1*5*math.pi/4, math.pi/2-thetas))
    jones = hp.getJonesAlong([frequency], (phis+1*5*math.pi/4, thetas))
    EsTh = numpy.squeeze(jones[...,0,0])
    EsPh = numpy.squeeze(jones[...,0,1])
    plt.subplot(2,1,1)
    plt.plot(phis/math.pi*180, 90-thetas/math.pi*180, '*')
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    plt.subplot(2,1,2)
    plt.plot(timeAngs*60, numpy.abs(EsTh))
    plt.xlabel('Transit time [min]')
    plt.ylabel('Gain [rel.]')
    plt.show()


def getJones(freq, az, el):
    """Print the Jones matrix of the HA model for a frequency and direction."""
    hp = HamakerPolarimeter(HA_LBAfile_default)
    jones=hp.getJonesAlong([10.e6], (0.1, 0.2))
    print "Jones:"
    print jones
    print "J.J^H:"
    print numpy.dot(jones, jones.conj().transpose()).real
    IXRJ = dualpolelem.getIXRJ(jones)
    print "IXRJ:", 10*numpy.log10(IXRJ),"[dB]"


def _getargs():
    freq = float(sys.argv[1])
    az = float(sys.argv[2])
    el = float(sys.argv[3])
    return freq, az, el


if __name__ == "__main__":
    #plotElemPat(30e6)
    showAnomaly()
    #HBAmod = HamakerPolarimeter(HA_HBAfile_default)
    #jones = HBAmod.getJonesAlong([150e6, 160e6, 170e6], ( [0.1,0.1], [0.3, 0.4]) )
    #print jones
