"""Vector Spherical Harmonics module. Based on my matlab functions."""
#TobiaC 2015-07-25
import sys
import math
import cmath
import scipy.special
import numpy


class Coefs(object):
  """Class for Vector Spherical Harmonics expansion Coefficients.
  Default format for coefficients is s,n,m tuple:
    coefs_snm[s][n][m] = [ [ [m=0,m=1,...,m=mMax,m=-mMaxm,...m=-1], [ ] ], [ [],[] ] ]
  """
  def __init__(self, *args, **kwargs):
    if len(args)==0:
      self.initBysnm( [ [[]],[[]] ])
    elif len(args)==1:
      self.initBysnm( args[0] )
    elif len(args)==2:
      self.initByQ1Q2( args[0], args[1] )
    #self.initByVec(coefVec)
    self.nrCoefs=self.CountNrCoefs()
    
  def __repr__(self):
    return str(self.coefs_snm)
    
  def __iter__(self):
    self.jIter=0
    return self
  
  def __call__(self, x):
    coefs_x=Coefs(self.coefs_snm)
    for snm in self:
      coefs_x.setBysnmt(snm, numpy.asscalar(self.getBysnmt(snm)(x)))
    return coefs_x
  
  def next(self):
    if self.jIter==self.nrCoefs:
      raise StopIteration
    self.jIter=self.jIter+1
    (s,n,m)=Coefs.j2snm(self.jIter)
    #return self.getBysnm(s,n,m)
    return (s,n,m)
  
  def CountNrCoefs(self):
    nr=0
    for el0 in self.coefs_snm:
      for el1 in el0:
        for el2 in el1:
          nr=nr+1
    return nr
  
  def initBysnm(self,coefs_snm):
    self.coefs_snm=coefs_snm
    self.LMAX=len(self.coefs_snm[0])
  
  def initByQ1Q2(self,Q1,Q2):
    self.LMAX=Q1.shape[1]
    self.setZeros(self.LMAX)
    for Li in range(self.LMAX):
      N=Li+1
      for Mi in range(2*N+1):
        M=Mi-Li-1
        Mind=self.LMAX-(Li+1)+Mi
        self.setBysnm(1,N,M,Q1[Mind,Li])
        self.setBysnm(2,N,M,Q2[Mind,Li])
  
  def initByVec(self,coefVec):
    self.coeftype = '1d'
    self.coefVec=coefVec
  
  @staticmethod
  def j2snm(j):
    s=2-(j % 2)
    n=int(math.floor(math.sqrt((j-s)/2+1)))
    m=(j-s)/2+1-n*(n+1)
    return s,n,m
  
  def getQ1Q2(self,Linear=False,EMmode=False,):
    Q1=numpy.zeros((2*self.LMAX+1,self.LMAX),dtype=complex)
    Q2=numpy.zeros((2*self.LMAX+1,self.LMAX),dtype=complex)
    for Li in range(self.LMAX):
      for Mi in range(2*(Li+1)+1):
        Mind=self.LMAX-(Li+1)+Mi
        N=Li+1
        M=Mi-Li-1
        Q1[Mind,Li]=self.getBysnm(1,N,M)
        Q2[Mind,Li]=self.getBysnm(2,N,M)
    if Linear:
      qsh=Q1.shape
      l0ind=int(numpy.floor((qsh[0]-1)/2.0))
      negmmask=numpy.ones(qsh)
      negmmask[0:l0ind,:]=-1
      Q1=(Q1+negmmask*numpy.flipud(Q1))/2.0
      negmmask[l0ind,:]=-1
      Q2=(Q2-negmmask*numpy.flipud(Q2))/2.0
    if EMmode :
      Qup=(Q1+1.0*Q2)/2.0
      Qdo=(Q1-1.0*Q2)/2.0
      Q1=Qup
      Q2=Qdo
    return Q1,Q2
  
  def getBysnm(self,s,n,m):
    si=s-1 #s is 1,2
    ni=n-1 #n is 1,2,3,...,Lmax for vector spherical harmonics.
    mi = m if m>=0 else 2*n+1+m
    return self.coefs_snm[si][ni][mi]
  
  def getBysnmt(self,snm):
    return self.getBysnm(snm[0],snm[1],snm[2])
  
  def setZeros(self,Lmax):
    coefs_snm=[]
    for si in range(2):
      coef__nm=[]
      for ni in range(Lmax):
        coef___m=[]
        for mi in range(2*(ni+1)+1):
          coef___m.append(0.0j)
        coef__nm.append(coef___m)
      coefs_snm.append(coef__nm)
    self.coefs_snm=coefs_snm
    self.nrCoefs=self.CountNrCoefs()
    self.LMAX=Lmax
  
  def setBysnm(self,s,n,m,cval):
    si=s-1 #s is 1,2
    ni=n-1 #n is 1,2,3,...,Lmax for vector spherical harmonics.
    mi = m if m>=0 else 2*n+1+m
    self.coefs_snm[si][ni][mi]=cval
  
  def setBysnmt(self,snm, cval):
    self.setBysnm(snm[0],snm[1],snm[2],cval)


def load_SWE_diag(sphfilename):
#Reads TICRA .sph files and returns SWE coef in my diagonal format.
  fp=open(sphfilename,'r')
  head1=fp.readline()
  head2=fp.readline()
  (NTHE,NPHI,MMAX,NMAX,bla)=[int(el) for el in fp.readline().strip().split()]
  Qd1=numpy.zeros((MMAX+1,MMAX+1),dtype=complex)
  Qd2=numpy.zeros((MMAX+1,MMAX+1),dtype=complex)
  headfreq=fp.readline().strip()
  head5=fp.readline().strip()
  head6=fp.readline().strip()
  blank=fp.readline().strip()
  blank=fp.readline().strip()
  Mind=0
  Mabs_pw=fp.readline().strip()
  for Nind in range(1,NMAX+1):
      (Q10r,Q10i,Q20r,Q20i)=[float(el) for el in fp.readline().strip().split()]
      Qd1[Nind,Nind]=complex(Q10r,Q10i)
      Qd2[Nind,Nind]=complex(Q20r,Q20i)
  for Mind in range(1,MMAX+1):
    Mabs_pw=fp.readline().strip().split()
    Mabs=int(Mabs_pw[0])
    pw=float(Mabs_pw[1])
    for Nind in range(Mind,NMAX+1):
      (Q1mr,Q1mi,Q2mr,Q2mi)=[float(el) for el in fp.readline().strip().split()]
      Qd1[Nind,Nind-Mind]=complex(Q1mr,Q1mi)
      Qd2[Nind,Nind-Mind]=complex(Q2mr,Q2mi)
      (Q1pr,Q1pi,Q2pr,Q2pi)=[float(el) for el in fp.readline().strip().split()]
      Qd1[Nind-Mind,Nind]=complex(Q1pr,Q1pi)
      Qd2[Nind-Mind,Nind]=complex(Q2pr,Q2pi)
  fp.close()
  return (Qd1,Qd2)


def load_SWE2vshCoef(sphfilename, convention='SWE'):
#Reads TICRA .sph files and returns SWE coef in Q1Q2 format.
  fp=open(sphfilename,'r')
  head1=fp.readline()
  head2=fp.readline()
  (NTHE,NPHI,MMAX,NMAX,bla)=[int(el) for el in fp.readline().strip().split()]
  Q1=numpy.zeros((2*MMAX+1,NMAX),dtype=complex)
  Q2=numpy.zeros((2*MMAX+1,NMAX),dtype=complex)
  headfreq=fp.readline().strip() # Frequency =   5.50000E+007 Hz
  frequency=float(headfreq.split('=')[1].strip().split()[0])
  head5=fp.readline().strip()
  head6=fp.readline().strip()
  blank=fp.readline().strip()
  blank=fp.readline().strip()
  Mind=0
  Mabs_pw=fp.readline().strip()
  for Nind in range(0,NMAX):
      (Q10r,Q10i,Q20r,Q20i)=[float(el) for el in fp.readline().strip().split()]
      Q1[NMAX,Nind]=complex(Q10r,Q10i)
      Q2[NMAX,Nind]=complex(Q20r,Q20i)
  for Mind in range(1,MMAX+1):
    Mabs_pw=fp.readline().strip().split()
    Mabs=int(Mabs_pw[0])
    pw=float(Mabs_pw[1])
    for Nind in range(Mind-1,NMAX):
      (Q1mr,Q1mi,Q2mr,Q2mi)=[float(el) for el in fp.readline().strip().split()]
      Q1[NMAX-Mind,Nind]=complex(Q1mr,Q1mi)
      Q2[NMAX-Mind,Nind]=complex(Q2mr,Q2mi)
      (Q1pr,Q1pi,Q2pr,Q2pi)=[float(el) for el in fp.readline().strip().split()]
      Q1[NMAX+Mind,Nind]=complex(Q1pr,Q1pi)
      Q2[NMAX+Mind,Nind]=complex(Q2pr,Q2pi)
  fp.close()
  #Postprocessing of GRASP .sph file coef to other coefficient convention
  if convention=='SWE':
    pass #default
  elif convention=='FEKO':
    # FEKO 7.0 manual:
    #   1) additional 1/sqrt(8*pi) factor
    #   2) coefficients are complex conjugate due to .sph having exp(-i*omega*t)
    #   3) m is exchanged with -m
    #In addition to conform with Hansen's K functions one needs to flip signs for m.
    #FEKO 7.0 sect 14.20 mentions (-1)^m not included.
    #Also because the j_n functions for FF have exp(ikr) dep in Hansen
    #while FEKO has opposite sign, s=1 should have factor -1^(n+1) and s=2 fac -1^n.
    Mvec=numpy.arange(0,Q1.shape[0])-NMAX #vector=[-m...0..m]
    Lvec=numpy.arange(1,NMAX+1)
    Mmask=numpy.outer(
            numpy.power(-1,Mvec),
            numpy.ones((Q1.shape[1],))
            ) # (-1)^m
    Lmask=numpy.power.outer(
            -numpy.ones((Q1.shape[0],)),Lvec+1
            ) # (-1)^(l+1)
    #NO L or M mask:
    #Mmask=numpy.ones(Q1.shape)
    #Lmask=+1*numpy.ones(Q1.shape)
    #print Lmask
    FEKOnrm=math.sqrt(8*math.pi)
    Q1=+Lmask*Mmask*numpy.flipud(numpy.conj(FEKOnrm*Q1))
    #Originally -1 due to factor -1^l rather than -1^(l+1)
    Q2=-Lmask*Mmask*numpy.flipud(numpy.conj(FEKOnrm*Q2))
    #Although point 3) above is stated in FEKO 7.0 manual, it does not seem to
    #actually be implemented... The following just implements 1) & 2).
    #FEKOnrm=1
    #Q1=numpy.conj(FEKOnrm*Q1)
    #Q2=numpy.conj(FEKOnrm*Q2)
    
  else:
    print ("Definition of output", outDef, " not known.")
    exit(1)
    
  Q12coef=Coefs(Q1,Q2)
  return Q12coef,frequency


def load_SWE2vshCoef_withFreqDep(sphfilenamelist):
  freqs=[]
  Q12coefFreq=[]
  for sphfilename in sphfilenamelist:
    Q12coef,frequency=load_SWE2vshCoef(sphfilename)
    Q12coefFreq.append(Q12coef)
    freqs.append(frequency)
  return Q12coefFreq, freqs


def load_SWE2vshField(sphfilenamelist):
   Q12coefFreq, freqs=load_SWE2vshCoef_withFreqDep(sphfilenamelist)
   return vshField(Q12coefFreq, freqs)
