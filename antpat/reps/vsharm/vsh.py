"""Vector Spherical Harmonics module. Based on my matlab functions."""
#TobiaC 2015-07-25
import sys
import math
import cmath
import scipy.special
import numpy


def Psi(l,m,theta,phi):
  """Computes the components of the zenithal Vector Spherical Harmonic function
  with l and m quantal numbers in the theta,phi direction."""
  if numpy.isscalar(theta): 
    theta=numpy.array([[theta]])
    phi=numpy.array([[phi]])
  Psilm_th=numpy.zeros(theta.shape,dtype=complex)
  Psilm_ph=numpy.zeros(theta.shape,dtype=complex)
  x=numpy.cos(theta)
  thetaNonZerosIdx=numpy.where(theta!=0.0)
  if len(thetaNonZerosIdx[0]) != 0:
    Ylm=scipy.special.sph_harm(m,l,phi[thetaNonZerosIdx],theta[thetaNonZerosIdx])
    #Compute derivative of sphrHarm function w.r.t. theta:
    if l>=numpy.abs(m):
      Plmpo=legendreLM(l,m+1,x[thetaNonZerosIdx])
      YlmPmpo=math.sqrt((2*l+1)/(4*math.pi)*math.factorial(l-m)/float(math.factorial(l+m)))*Plmpo*numpy.exp(1j*m*phi[thetaNonZerosIdx])
      #YlmPmpo=sqrt((l-m)*(l+m+1))*spharm(l,m+1,theta,phi)*exp(-i*phi) %Should be equivalent to above formula.
      dtYlm=+YlmPmpo+m*x[thetaNonZerosIdx]*Ylm/numpy.sin(theta[thetaNonZerosIdx])
      #  thetZerInd=[find(theta==0); find(theta==pi)]
      #  dtYlm(thetZerInd)=0; %This is a fudge to remove NaNs
    else:
      dtYlm=numpy.zeros(theta[thetaNonZerosIdx].shape,dtype=complex)

      #dtYlm=spharmDtheta(l,m,theta,phi)

    Psilm_ph[thetaNonZerosIdx]=+1j*m/numpy.sin(theta[thetaNonZerosIdx])*Ylm
    Psilm_th[thetaNonZerosIdx]=+dtYlm
    #Ref: http://mathworld.wolfram.com/VectorSphericalHarmonic.html

  thetaZerosIdx=numpy.where(theta==0.0)
  if len(thetaZerosIdx[0]) != 0:
    if numpy.abs(m)==1:
      Yl1B=math.sqrt((2*l+1)/(4*math.pi)*math.factorial(l-m)/math.factorial(l+m))*PBl1(l,m)*numpy.exp(1j*m*phi[thetaZerosIdx])
      Plmpo=legendreLM(l,m+1,x[thetaZerosIdx])
      YlmPmpo=math.sqrt((2*l+1)/(4*math.pi)*math.factorial(l-m)/math.factorial(l+m))*Plmpo*numpy.exp(1j*m*phi[thetaZerosIdx])
      dtYlm=+YlmPmpo+m*Yl1B
      Psilm_ph[thetaZerosIdx]=+1j*m*Yl1B
      Psilm_th[thetaZerosIdx]=+dtYlm
    else:
        Plmpo=legendreLM(l,m+1,x[thetaZerosIdx])
        YlmPmpo=math.sqrt((2*l+1)/(4*math.pi)*math.factorial(l-m)/math.factorial(l+m))*Plmpo*numpy.exp(1j*m*phi[thetaZerosIdx])
        dtYlm=+YlmPmpo+0
        Psilm_ph[thetaZerosIdx]=0
        Psilm_th[thetaZerosIdx]=+dtYlm
  return Psilm_th,Psilm_ph


def Phi(l,m,theta,phi):
  """Computes the components of the azimuthal Vector Spherical Harmonic function
  with l and m quantal numbers in the theta,phi direction."""
  Psilm_th, Psilm_ph=Psi(l,m,theta,phi);
  Philm_th=-Psilm_ph;
  Philm_ph=+Psilm_th;
  return Philm_th, Philm_ph


def K(s,n,m,theta,phi):
#Hansen1988 far-field functions. Needs to be corrected.
  if s==1:
    K_th,K_ph= Psi(n,m,theta,phi)
  elif s==2:
    K_th,K_ph=Phi(n,m,theta,phi)
  return K_th, K_ph


def legendreLM(l,m,x):
    lout=scipy.special.lpmv(abs(m),l,x)
    if m<0:
       m=abs(m)
       lout=(-1)**m*math.factorial(l-m)/math.factorial(l+m)*lout
    return(lout)


def PBl1(l_target, m):
  """Compute the value of the apparent singularity for a function within the
  VSH at the north pole. The function is P_l^m(x)/\sqrt{1-x^2}
  where P_l^m(x) is the associate Legendre polynomial. It is evaluated
  for x=+1 and |m|=1. (For |m|>1 this function is 0 at x=+1.)
  (This function is called by vshPhi and vshPsi)"""
  #TobiaC 2011-10-13 (2011-10-13)
  PBtab=numpy.array([0.0, -1.0, -3.0])
  l=2
  while l_target>l:
    PBlp1=((2*l+1)*PBtab[l +0]-(l+1)*PBtab[l-1 +0])/float(l)
    PBtab=numpy.hstack((PBtab, numpy.array([PBlp1])))
    l=l+1
  if m==1:
    PBout=PBtab[l_target +0]
  else:
    if m==-1:
      PBout=-math.factorial(l_target-1)/float(math.factorial(l_target+1))*PBtab[l_target +0]
    else:
          PBout=0.0
  return PBout


def vsfun(Q_slm, theta, phi,f=[]):
  """Direct computation of a vector function with spherical components theta,phi based on the
  vector spherical harmonics expansion with coefficients Q_slm."""
  vsf_th=numpy.zeros(theta.shape, dtype='complex')
  vsf_ph=numpy.zeros(theta.shape, dtype='complex')
  for (s,l,m) in Q_slm:
    vsh_th,vsh_ph=K(s, l, m, theta, phi)
    c_slm=Q_slm.getBysnm(s, l, m) if not(f) else Q_slm.getBysnm(s, l, m)(f)
    vsf_th=vsf_th+c_slm*vsh_th
    vsf_ph=vsf_ph+c_slm*vsh_ph
  return vsf_th, vsf_ph

