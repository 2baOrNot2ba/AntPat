#!/usr/bin/python
from vsh import *
from coefs import *
import matplotlib.pyplot as plt

def show_SWE(filename):
  Q12coef,freq=load_SWE2vshCoef(filename)
  Q1,Q2=Q12coef.getQ1Q2(Linear=False,EMmode=False)
  print (Q1)
  print (Q2)
  qsh=Q1.shape
  l0ind=int(numpy.floor((qsh[0]-1)/2.0))
  q1=numpy.log10(numpy.abs(Q1))
  #q1=numpy.abs(Q1)
  q2=numpy.log10(numpy.abs(Q2))
  #q2=numpy.abs(Q2)
  plt.matshow(q1,extent=[1-0.5,l0ind+0.5,l0ind+0.5,-l0ind-0.5])
  plt.title("|Q1|")
  plt.colorbar()
  plt.matshow(q2,extent=[1-0.5,l0ind+0.5,l0ind+0.5,-l0ind-0.5])
  plt.title("|Q2|")
  plt.colorbar()
  plt.show()

def testSWEfreq():
  Q12coefFreq, freqs=load_SWE2vshCoef_withFreqDep(sys.argv[1:])
  print (Q12coefFreq[0].coefs_snm)
  y=numpy.array([])
  for freqNr in range(len(freqs)):
    y=numpy.hstack((y,Q12coefFreq[freqNr].getBysnm(1,1,1)))
  plt.plot(freqs,numpy.abs(y),'rs')
  plt.show()


if __name__ == "__main__":
  #testPat()
  #testSingPat()
  #print Psi(1,-1,0.00,1.0)
  show_SWE(sys.argv[1])
  #c=Coefs([[[1.0,2.0,3.0],[7.0,8.0,0.0,9.0,10,0]],[[4.0,5.0,6.0],[7.0,8.0,0.0,9.0,10,0]]])
  #Q1,Q2=c.getQ1Q2()
  #print Q1
  #testSWEfreq()
