#!/usr/bin/python
# TobiaC 2015-11-25 (2013-07-11)
import sys
import math
import numpy
from antpat.reps.sphgridfun import tvecfun


def readNECout_tvecfuns(filename):
    """Read a NEC .out file and return it as a TVecField instance.
    """
    frequencies, thetaMsh, phiMsh, EthetaF, EphiF = readNECout_FF(filename)
    # Check to see if mesh wraps around in azimuth. If it does, remove the
    # redundant components. (It may have to been done elsewhere)
    if numpy.fmod(phiMsh[0, 0], 2*math.pi) \
       == numpy.fmod(phiMsh[0, -1], 2*math.pi):
        phiMsh = numpy.delete(phiMsh, (-1), axis=1)
        thetaMsh = numpy.delete(thetaMsh, (-1), axis=1)
        EthetaF = numpy.delete(EthetaF, (-1), axis=-1)
        EphiF = numpy.delete(EphiF, (-1), axis=-1)
    # End of wrap check
    FFs = tvecfun.TVecFields(thetaMsh, phiMsh, EthetaF, EphiF, frequencies)
    return FFs


def readNECout_FF(filename):
    """Read in a NEC .out file."""
    fp = open(filename, "r")
    dc = _readNECout_datacards(fp)
    nrThetas = int(dc['RP'][1])
    nrPhis = int(dc['RP'][2])
    nrFreqs = int(dc['FR'][1])
    frequencies = numpy.zeros((nrFreqs))
    fp.seek(0)
    FREQREP = 'ARRax0'
    if FREQREP == 'LIST':
        EthetaF = []
        EphiF = []
    elif FREQREP == 'ARRax0':
        EthetaF = numpy.zeros((nrFreqs, nrThetas, nrPhis), dtype=complex)
        EphiF = numpy.zeros((nrFreqs, nrThetas, nrPhis), dtype=complex)
    for freqInd in range(nrFreqs):
        frequency, thetaMsh, phiMsh, Etheta, Ephi = _readNECout_FFnextFreq(fp)
        frequencies[freqInd] = frequency
        if FREQREP == 'LIST':
            EthetaF.append(Etheta)
            EphiF.append(Ephi)
        else:
            EthetaF[freqInd, :, :] = Etheta
            EphiF[freqInd, :, :] = Ephi
    thetaMsh = thetaMsh*numpy.pi/180
    phiMsh = phiMsh*numpy.pi/180
    fp.close()
    return frequencies, thetaMsh, phiMsh, EthetaF, EphiF


def _readNECout_datacards(fp):
    fp.seek(0)
    datacards = {}
    datacardsIdx = {}
    datacardCardinal = 0
    datacard_markers = ['DATA CARD No:', '***** INPUT LINE']
    for line in fp:
        if any(marker in line for marker in datacard_markers):
            line = line.lstrip()
            for marker in datacard_markers:
                line = line.lstrip(marker)
            line = line.rstrip()
            datacardCardinal += 1
            dcparams = line
            dcNr, dcType, dcTypeParams = dcparams.split(None, 2)
            datacardsIdx[dcNr] = datacardCardinal
            datacards[dcType] = dcTypeParams.split()
    return datacards


def _readNECout_FFnextFreq(fp):
    thetaList = []
    phiList = []
    EthetaMagList = []
    EthetaPhsList = []
    EphiMagList = []
    EphiPhsList = []

    while True:
        aline = fp.readline()
        if "- FREQUENCY -" in aline:
            freqline = ""
            while not freqline:
                freqline = fp.readline().strip()
            freqlinelst = freqline.split()
            freqStr, freqUnit = freqlinelst[-2:]
            freq = float(freqStr)
            if freqUnit == 'Hz':
                freq = freq*1
            elif freqUnit == 'kHz':
                freq = freq*1e3
            elif freqUnit == 'MHz':
                freq = freq*1e6
            elif freqUnit == 'THz':
                freq = freq*1e9
            break

    while True:
        aline = fp.readline()
        if "RADIATION PATTERNS" in aline:
            break
    patTabHdGrp = ""
    while not patTabHdGrp:
        patTabHdGrp = fp.readline().strip()
    patTabHdQnt = fp.readline()
    patTabHdUnt = fp.readline()
    while True:
        aline = fp.readline().strip()
        if not aline or ('DATA CARD' in aline):
            break
        else:
            if ('LINEAR' in aline) or ('RIGHT' in aline) or ('LEFT' in aline):
                theta, phi, GainVert, GainHor, GainTot, PolAx, PolTlt, PolSns,\
                    EthetaMag, EthetaPhs, EphiMag, EphiPhs = aline.split()
            else:
                theta, phi, GainVert, GainHor, GainTot, PolAx, PolTlt,\
                    EthetaMag, EthetaPhs, EphiMag, EphiPhs = aline.split()
            thetaList.append(float(theta))
            phiList.append(float(phi))
            EthetaMagList.append(float(EthetaMag))
            EthetaPhsList.append(float(EthetaPhs))
            EphiMagList.append(float(EphiMag))
            EphiPhsList.append(float(EphiPhs))
    thetaVals = set(thetaList)
    phiVals = set(phiList)
    NrThetas = len(thetaVals)
    NrPhis = len(phiVals)
    thetaMsh = numpy.transpose(numpy.array(thetaList).reshape(NrPhis, NrThetas)
                               )
    phiMsh = numpy.transpose(numpy.array(phiList).reshape(NrPhis, NrThetas))
    EthetaA = numpy.transpose((numpy.array(EthetaMagList)
                               * numpy.exp(1j*numpy.deg2rad(
                                 numpy.array(EthetaPhsList))))
                              .reshape(NrPhis, NrThetas))
    EphiA = numpy.transpose((numpy.array(EphiMagList)
                             * numpy.exp(1j*numpy.deg2rad(
                               numpy.array(EphiPhsList))))
                            .reshape(NrPhis, NrThetas))
    return freq, thetaMsh, phiMsh, EthetaA, EphiA


if __name__ == "__main__":

    filename = sys.argv[1]
    frequencies, thetaMsh, phiMsh, Etheta, Ephi = readNECout_FF(filename)
    print(frequencies)
    print(Etheta.shape)
