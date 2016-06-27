"""Various theoretically interesting antenna types."""
import numpy
from .radfarfield import RadFarField
from .reps import sphgridfun
from .reps.vsharm import vsh
from .reps.vsharm.vshfield import vshField
from .reps.vsharm.coefs import Coefs as vshCoefs


def max_gain_pat(Nmax):
    """Gives a RadFarField instance of a maximum gain antenna. Ref Pozar2007"""
    cfs = vshCoefs()
    cfs.setZeros(Nmax)
    alpha = 1.0
    for n in range(1,Nmax+1):
      #RHCP
      a1 = (alpha*(2*n+1))*numpy.power(1j,n-1)/numpy.sqrt(n*(n+1)) #Extra factor *n*(n+1))
      c1 = a1
      cfs.setBysnm(1,n,1,a1)
      cfs.setBysnm(2,n,1,c1)
      #LHCP
      
    #print cfs
    vshcfs = vshField([cfs])
    return RadFarField(vshcfs)


def dirac_beam(freqs=[0.0], direction='X'):
    """Gives a RadFarField instance with radiation only a given direction."""
    freqs = numpy.asarray(freqs)
    theta, phi = sphgridfun.pntsonsphere.sphericalGrid(16,32)
    gridshape = theta.shape
    xindtheta = (gridshape[0]-1)/2
    freqshp=freqs.shape
    E_th = numpy.zeros(freqshp+gridshape)
    E_ph = numpy.zeros(freqshp+gridshape)
    if direction == 'X':
        E_th[:,xindtheta, 0] = 1.0
    diracpat = RadFarField(sphgridfun.tvecfun.TVecFields(theta, phi, E_th, E_ph, freqs))
    return diracpat


def ideal_dipole_grid_X(freqs):
    thetamsh, phimsh = sphgridfun.pntsonsphere.sphericalGrid(32, 64)
    E_thm1, E_phm1 = vsh.Psi(1, -1, thetamsh, phimsh)
    E_thp1, E_php1 = vsh.Psi(1, +1, thetamsh, phimsh)
    E_th = E_thm1+E_thp1
    E_ph = E_phm1+E_php1
    E_th = numpy.resize(E_th, freqs.shape+thetamsh.shape)
    E_ph = numpy.resize(E_ph, freqs.shape+thetamsh.shape)
    atvfd=sphgridfun.tvecfun.TVecFields(thetamsh, phimsh, E_th, E_ph, freqs)
    antFF=RadFarField(atvfd)
    return antFF
