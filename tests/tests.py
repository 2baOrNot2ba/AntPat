#!/usr/bin/python
"""Test scripts for antpat package."""
import sys
import os
import math
import numpy
import matplotlib
matplotlib.use('WXAgg') #This one works
#matplotlib.use('GTKCairo')
import matplotlib.pyplot as plt
from antpat.radfarfield import RadFarField
from antpat.io import NECread
from antpat.reps.vsharm import vsh, hansens
from antpat.reps.vsharm.coefs import Coefs as vshCoefs
from antpat.reps.vsharm.coefs import load_SWE2vshCoef
from antpat.reps.vsharm.vshfield import vshField
from antpat.reps.sphgridfun import pntsonsphere, tvecfun
import antpat.theoreticalantennas
from antpat.dualpolelem import DualPolElem
from antpat.reps.hamaker import HamakerPolarimeter
#import antpat.gen1dfun.Pade as Pade


projdir = os.path.dirname(os.path.abspath('.'))
dataformdir = projdir+'/example_FF_files/'
SWEdir = dataformdir+'/SWE/'
NECdir = dataformdir+'/NEC_out/'
FFEdir = dataformdir+'/FFE/'


def NECout():
  """Test read of NEC .out file."""
  basefile = 'Lofar-dipole-FREE.out'
  fileloc = NECdir+basefile
  atvfd = NECread.readNECout_tvecfuns(fileloc)
  antFF = RadFarField(atvfd)
  return antFF

def NECplot3D():
  """Test plot NEC 3D ant pattern"""
  antFF = NECout()
  selfreq = 110.0e6
  #antFF.plotAntPat3D(selfreq)
  THETA, PHI, E_th, E_ph = antFF.getFFongrid(selfreq)
  tvecfun.plotvfonsph3D(THETA, PHI, E_th, E_ph)

def NECplotspec():
  """Test spec"""
  antFF = NECout()
  frequencies = antFF.getfreqs()
  nrFreqs = len(frequencies)
  theta_phi = (numpy.array([0.1,0.2]),numpy.array([0.5,0.6]))
  EF = antFF.getFFalong(frequencies, theta_phi)
  plt.figure()
  plt.pcolormesh(numpy.abs(EF[0]))
  plt.show()

def vsh_pat_freq():
  """Test plot of VSH module with frequency dependence"""
  theta = numpy.linspace(0.0,math.pi,num=100)
  phi = numpy.linspace(0.0,2*math.pi,num=100)
  PHI, THETA = numpy.meshgrid(phi,theta)
  
  vc = vshCoefs()
  vc.setZeros(1)
  
  vc.setBysnm(1, 1,-1, Pade.Approximant([1.0],[1.0,-1.0+0.1j]).valAt)
  vc.setBysnm(1, 1, 0, Pade.Approximant([0.0],[1.0,-1.0+0.1j]).valAt)
  vc.setBysnm(1, 1, 1, Pade.Approximant([-1.0],[1.0,-1.0+0.0j]).valAt)
  vc.setBysnm(2, 1,-1, Pade.Approximant([1.0j],[1.0,-1.0+0.1j]).valAt)
  vc.setBysnm(2, 1, 0, Pade.Approximant([0.0],[1.0,-1.0+0.1j]).valAt)
  vc.setBysnm(2, 1, 1, Pade.Approximant([1.0j],[1.0,-1.0+0.1j]).valAt)
  
  E_th, E_ph = vsh.vsfun(vc, THETA, PHI, f=0.1)
  tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph)


def gen_simp_pat():
  """Generate a linear y directed dipole."""
  THETA, PHI = pntsonsphere.sphericalGrid(100, 100)
  E_thm1, E_phm1 = vsh.Psi(1, -1, THETA, PHI)
  E_thp1, E_php1 = vsh.Psi(1, +1, THETA, PHI)
  E_th = E_thm1+E_thp1
  E_ph = E_phm1+E_php1
  return THETA, PHI, E_th, E_ph

def gen_simp_RadFarField():
    THETA, PHI, E_th, E_ph = gen_simp_pat()
    atvfd = tvecfun.TVecFields(THETA, PHI, E_th, E_ph, 1.0)
    antFF = RadFarField(atvfd)
    return antFF

def plot_simp_pat2D():
    """Test 2D plot of the tangential vector function given by the
    spherical harmonic function Psi in vsh package."""
    THETA, PHI, E_th, E_ph = gen_simp_pat()
    print E_th
    tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph)

def plot_simp_pat3D():
    """Test 3D plot of the tangential vector function given by the
    spherical harmonic function Psi in vsh package."""
    THETA, PHI, E_th, E_ph = gen_simp_pat()
    tvecfun.plotvfonsph3D(THETA, PHI, E_th, E_ph)

def vshcoefs():
    """Test plot of a tangential vector function given by vsh
    coefficients."""
    theta = numpy.linspace(0.0, math.pi, num=32)
    phi = numpy.linspace(0.0, 2*math.pi, num=32)
    PHI, THETA = numpy.meshgrid(phi, theta)
    cfl = [[[0.0,1.0,-1.0]],[[0.0,0*1.0j,0*1.0j]]]
    cfs = vshCoefs(cfl)
    E_th, E_ph = vsh.vsfun(cfs,THETA,PHI)
    tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph,
                      vcoord='sph',projection='equirectangular')
    #tvecfun.plotvfonsph3D(THETA, PHI, E_th, E_ph)

def hansen():
  """Test plot of a radiation far-field given by vsh coefficients."""
  theta = numpy.linspace(0.0, math.pi, num=100)
  phi = numpy.linspace(0.0, 2*math.pi, num=200)
  PHI, THETA = numpy.meshgrid(phi, theta)
  cfl=[[[0.0, 1*1.0, -1],[0.0, 0*1j*1.0, 0.0, 0.0, 0.0],[0.0, 0*-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
      ,[[0.0, 0*1.0, 0.0],[0.0, 0*1j*1.0, 0.0, 0.0, 0.0],[0.0, 0*-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
  cfs = vshCoefs(cfl)
  cfsl = vshField([cfs], [0.0])
  antFF = RadFarField(cfsl)
  antFF.plotAntPat3D(vcoord='Ludwig3',projection='equirectangular')

def vshreal():
  """Test plot of a radiation far-field given by vsh coefficients taken
  from a SWE file."""
  basefile = 'Ideal_DipTE_m1.sph'
  SWEfileloc = SWEdir+basefile
  cfs, freq = load_SWE2vshCoef(SWEfileloc, convention='FEKO')
  print(basefile)
  print(freq)
  antFF = RadFarField(vshField([cfs], [freq]))
  THETA, PHI, V_th, V_ph = antFF.getFFongrid(freq)
  c = 3.0e8
  k = (2*math.pi*freq)/c
  Z0 = 376.7
  V2EfieldNrm = k*numpy.sqrt(Z0/(2*2*math.pi)) #Not sure about a 1/sqrt(2) factor
  E_th = V2EfieldNrm*V_th
  E_ph = V2EfieldNrm*V_ph
  #plotAntPat2D(theta_var,phi_fix,E_th,E_ph)
  tvecfun.plotvfonsph(THETA, PHI, E_th, E_ph, freq,
                      vcoord='sph',projection='equirectangular')

def maxgainpat(Nmax=10):
    """Test plot of a theoretical antenna that has a max gain pattern."""
    mxGant = antpat.theoreticalantennas.max_gain_pat(Nmax)
    THETA, PHI, E_th, E_ph = mxGant.getFFongrid(0.0)
    tvecfun.plotvfonsph3D(THETA, PHI, E_th, E_ph)

def test_diracb():
    """Test Dirac beam function."""
    dbant = antpat.theoreticalantennas.dirac_beam()
    dbant.plotAntPat3D()

def passrotPat():
    """Test passive rotation of a pattern. There are two sub-requests possible
    and two input configs: full 3D output of 2D-cuts with input, single-pol
    pattern or dual-pol.
    """
    def doTrack():
        #(thetas, phis) = pntsonsphere.cut_az(0.*math.pi/2) #Good for some tests.
        (thetas, phis) = pntsonsphere.cut_theta(10.0/180*math.pi)
        try:
            E_ths, E_phs = ant.getFFalong(1.0, (thetas, phis))
            tvecfun.plotAntPat2D(thetas, E_ths, E_phs, freq=0.5)
        except:
            freq=30e6
            jones = ant.getJonesAlong([freq], (thetas, phis))
            j00=jones[...,0,0].squeeze()
            j01=jones[...,0,1].squeeze()
            tvecfun.plotAntPat2D(phis, j00, j01, freq)
            j10=jones[...,1,0].squeeze()
            j11=jones[...,1,1].squeeze()
            tvecfun.plotAntPat2D(phis, j10, j11, freq)
        
    def do3D():
        cutphis = numpy.arange(0, 2*math.pi, .2)
        nrLngs = len(cutphis)
        dims = (100, nrLngs)
        THETA = numpy.zeros(dims)
        PHI  = numpy.zeros(dims)
        E_TH = numpy.zeros(dims, dtype=complex)
        E_PH = numpy.zeros(dims, dtype=complex)
        for (cutNr, cutphi) in enumerate(cutphis):
            (thetas, phis) = pntsonsphere.cut_theta(cutphi)
            E_ths, E_phs = ant.getFFalong(0.0, (thetas, phis))
            THETA[:,cutNr] = thetas
            PHI[:,cutNr] = phis
            E_TH[:,cutNr] = E_ths
            E_PH[:,cutNr] = E_phs
        tvecfun.plotvfonsph(THETA, PHI, E_TH, E_PH, projection='equirectangular')
    
    #Get a simple linear dipole along y.
    singpol = True
    if singpol:
        #ant = gen_simp_RadFarField()
        ant = antpat.theoreticalantennas.max_gain_pat(4)
    else:
        ha = HamakerPolarimeter('HA_LOFAR_elresp_LBA.p')
        ant = DualPolElem(ha)
    
    rotang = 1.*math.pi/4.
    rotmat = pntsonsphere.rot3Dmat(0.0, 0.3*math.pi/2, 0.1*math.pi/2)
    #Rotate the antenna 90 deg.
    print(rotmat)
    ant.rotateframe(rotmat)
    #Choose between next 2 lines:
    #doTrack()
    do3D()

def dualpolelem_2FF():
    """Test plot of a dual-polarized antenna where one channel is given
    by Psi and the other is a rotated copy it."""
    T, P = pntsonsphere.sphericalGrid()
    dipX = numpy.array(vsh.Psi(1, 1, T, P))+numpy.array(vsh.Psi(1, -1, T, P))
    dipXT = numpy.squeeze(dipX[0,:,:])
    dipXP = numpy.squeeze(dipX[1,:,:])
    freq = 1.0
    vfdipX = tvecfun.TVecFields(T, P, dipXT, dipXP, freq)
    antp = RadFarField(vfdipX)
    antq = antp.rotate90()
    dualpolAnt = DualPolElem(antp, antq)
    rotmat = pntsonsphere.rotzmat(0*math.pi/4)
    dualpolAnt.rotateframe(rotmat)
    dualpolAnt.plotJonesPat3D(freq, projection='azimuthal-equidistant', cmplx_rep='ReIm', )

if __name__ == "__main__":
    testfuns=filter(callable, locals().values())
    print("---List of function tests---")
    for testfun in testfuns:
        testfunname=testfun.__name__
        thedocstr=testfun.__doc__
        if thedocstr is None:
            print(testfunname+" : N/A")
        else:
            thedoclines=thedocstr.split()
            mxlinlen=50
            sys.stdout.write(testfunname+" : ")
            linlen=0
            for wrd in thedoclines:
                sys.stdout.write(wrd+" ")
                linlen+=len(wrd)+1
                if linlen>mxlinlen:
                    linlen=0
                    sys.stdout.write('\n        ')
            sys.stdout.write('\n')
            #thedoclines=[thedocstr[li:li+linlen] for li in range(0,len(thedocstr),linlen)]
            #print(testfunname+" : "+thedoclines[0])
            #for li in range(1,len(thedoclines)):
            #    print(thedoclines[li])
    sel=raw_input('Which one? ')
    eval(sel+'()')
