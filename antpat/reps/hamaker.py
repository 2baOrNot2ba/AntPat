#!/usr/bin/python
"""Hamaker's analytic antenna pattern model."""
# TobiaC 2015-11-29 (2015-07-31)

import sys
import math
import re
import pickle
import numpy
import antpat
from antpat import dualpolelem, radfarfield
from antpat.reps.sphgridfun import tvecfun, pntsonsphere
HA_LBAfile_default = ''
PICKLE_PROTO = pickle.HIGHEST_PROTOCOL


class HamakerPolarimeter(object):
    """This is the Hamaker polarimeter model class.
    It is the default LOFAR pipeline model for LOFAR beams.

    Note
    ----
    The Hamaker model suffers from two main problems:
        1) In polar angle it is best near the pole (boresight) and not so good
           away from it
        2) Discontinuities at the pole (boresight) can arise if coefficients
           with ki>0, ti=0 are nonzero.
    """

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

        Note
        ----
        Formula used here is based on the Hamaker model for crossed dipole
        pairs.

        P[comp, ki, freq, theta] = \
                            sum_ti, fi coefs[ki, ti, fi, comp]*freq^fi*theta^ti

        J[freq, theta_phi, pol, comp] = \
                    sum_ki R(ang(ki,phi))[pol, comp] * P[comp, ki, freq, theta]

        R(ki*phi) = [cos(ang(ki,phi))  -sin(ang(ki,phi))
                     sin(ang(ki,phi))   cos(ang(ki,phi))]

        ang(ki,phi) = (-1)^ki*(2*ki+1)*phi

        Parameters
        ----------
        freqvals : list
            list of frequencies in Hz.
        theta_phi : tuple
            tuple, with first element an array of theta, and second element
            array of phi, both in radians.

        Returns
        -------
        response : ndarray
            Jones matrix over frequencies and directions. The indices are
                response[freq, dir, polchan, comp] where
                    freq is frequency,
                    dir is a theta, phi direction,
                    polchan is polarization channel,
                    comp is component.
        """
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

    def scale(self, scalefac):
        """Scale Hamaker model by a multiplicative factor scalefac."""
        self.coefs = scalefac*self.coefs

    def _getJonesAlong_alt(self, freqvals, theta_phi):
        """Alternative calculation of JonesAlong using _basefunc."""
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
        res = numpy.zeros(frqXdrn_shp+(2, 2), dtype=complex)
        for ki in range(k_ord):
            for THi in range(TH_ord):
                for FRi in range(FR_ord):
                    bf = _basefunc(ki, THi, FRi, freqn, theta_phi)
                    res[..., 0, 0] += self.coefs[ki, THi, FRi, 0]*bf[0, ...]
                    res[..., 0, 1] += self.coefs[ki, THi, FRi, 1]*bf[1, ...]
                    res[..., 1, 0] += self.coefs[ki, THi, FRi, 0]*(-bf[1, ...])
                    res[..., 1, 1] += self.coefs[ki, THi, FRi, 1]*bf[0, ...]
        return res


def _basefunc(ki, ti, fi, frqn, theta_phi):
    """Computes a basis function for Hamaker expansion.

    A Hamaker basis function is a monomial in theta, freqn multiplied by
    a sinusoid in phi. The order of the monomial is given by fi, ti and
    the sinusoid is order is given by ki. The X-directed Hamaker basis
    functions are:
        ham_X[0, frqn_idx, theta_phi_idx] = +cos(ang)*(frqn**fi)*(tht**ti)
        ham_X[1, frqn_idx, theta_phi_idx] = -sin(ang)*(frqn**fi)*(tht**ti)
    where
        ang = (-1)**ki*(2*ki+1)*phi
    and
        tht, phi = theta_phi
        0, 1: are components of incoming field components.

    There is also an additional pair of Hamaker basis functions for
    Y-directed dipoles:
       ham_Y[0, ...] = -ham_X[1, ...]
       ham_Y[1, ...] = +ham_X[0, ...]

    Note
    ----
    Hamaker

    Parameters
    ----------
    ki : int
        Order of phi in basis function.
    ti : int
        Order of theta in basis function.
    fi : int
        Order of frqn in basis function.
    frqn : array_like
        Normalized (interval=(-1, 1)) array of real frequencies.
    theta_phi : tuple
        (tht, phi) where tht is an array of theta and phi is array of azi.

    Returns
    -------
    ham_x : array
        Hamaker X basis function with ham_X[comp, frqn_idx, theta_phi_idx].
    """
    tht, phi = theta_phi
    tht = numpy.array(tht)
    phi = numpy.array(phi)
    fac = numpy.multiply.outer(frqn**fi, tht**ti)
    ang = (-1)**ki*(2*ki+1)*phi
    r_x = numpy.array([+numpy.cos(ang), -numpy.sin(ang)])
    r_x = r_x[:, numpy.newaxis, ...]
    # fac:       frqord x thphord
    # r_x:   2 x   1    x thphord
    #  * :   --------------------
    # ham_x: 2 x frqord x thphord
    ham_x = fac*r_x
    return ham_x


def hamaker_coefs(patobj, freq_center, freq_range, kord=2, tord=5, ford=5):
    """Estimate the coefficients of the Hamaker for a far-field pattern given
    by patobj. One should specify the frequency center [Hz], the frequency
    range [Hz] and the order of the azimuthal (kord), theta angle (tord) and
    frequency (ford).

    Note
    ----
    One should try to put the freq_center value as close the global maximum
    of the frequency response, i.e. it should the frequency where the antenna
    has a resonance.

    Returns a numpy array of complex coefficients indexed as
        coefs[k][t][f][p]
    with shape (kord, tord, ford, 2). These can then be used in the ArtData
    dictionary in the specification of the HamakerPolarimeter class.
    """
    from antpat.reps.sphgridfun.pntsonsphere import ZenHemisphGrid
    from numpy.linalg import lstsq
    nfreq, ntheta, nphi = ford*3, tord*3, kord*4
    freqsmp = numpy.linspace(freq_center-freq_range, freq_center+freq_range,
                             nfreq)
    freqsnrm = (freqsmp - freq_center)/freq_range

    thetamsh, phimsh = ZenHemisphGrid(ntheta, nphi, incl_equator=False)
    if isinstance(patobj, tvecfun.TVecFields):
        Etheta, Ephi = patobj.getFalong(thetamsh, phimsh, freqsmp)
    elif isinstance(patobj, radfarfield.RadFarField) \
            or isinstance(patobj, dualpolelem.DualPolElem):
        Etheta, Ephi = patobj.getFFalong(freqsmp, (thetamsh, phimsh))
    ff0 = Etheta.flatten()
    ff1 = Ephi.flatten()
    bidx_shp = (kord, tord, ford)
    bidx_ord = numpy.prod(bidx_shp)
    ivar_shp = (nfreq, ntheta, nphi)
    ivar_ord = numpy.prod(ivar_shp)
    bfnd0 = numpy.zeros((ivar_ord, bidx_ord), dtype=float)
    bfnd1 = numpy.zeros((ivar_ord, bidx_ord), dtype=float)
    for ki in range(kord):
        for ti in range(tord):
            for fi in range(ford):
                ham_x = _basefunc(ki, ti, fi, freqsnrm, (thetamsh, phimsh))
                bidx_idx = numpy.ravel_multi_index(([ki], [ti], [fi]),
                                                   bidx_shp).squeeze()
                bfnd0[:, bidx_idx] = ham_x[0].flatten()
                bfnd1[:, bidx_idx] = ham_x[1].flatten()
    sol0 = lstsq(bfnd0, ff0)[0].reshape(bidx_shp)
    sol1 = lstsq(bfnd1, ff1)[0].reshape(bidx_shp)
    coefs = numpy.moveaxis(numpy.array([sol0, sol1]), 0, -1)
    return coefs


def _write_LOFAR_HAcc(artsdata):
    """Write Arts data to a LOFAR .cc file.
    The filename will be '<HAcoefversion>Coeff<HAcoefband>.cc',
    where <HAcoefversion> is the version name of the coefficients and
    <HAcoefband> is the band (typically LBA or HBA); both are keys in the
    artsdata dict argument.
    """
    coefs = artsdata['coefs']
    (kord, tord, ford, pord) = coefs.shape
    varprefix = "{}_{}".format(artsdata['HAcoefversion'],
                               artsdata['HAcoefband'].lower())
    filename = "{}Coeff{}.cc".format(artsdata['HAcoefversion'],
                                     artsdata['HAcoefband'].upper())
    with open(filename, 'w') as fp:
        fp.write("//Created by AntPat version {}\n".format(antpat.__version__))
        fp.write("#include <complex>\n")
        fp.write("const double {}_freq_center = {};\n".format(
            varprefix, artsdata['freq_center']))
        fp.write("const double {}_freq_range = {};\n".format(
            varprefix, artsdata['freq_range']))
        fp.write("const unsigned int {}_coeff_shape[3] = {{{}, {}, {}}};\
            \n".format(varprefix, kord, tord, ford))
        fp.write("const std::complex<double> {}_coeff[{}] = {{\
            \n".format(varprefix, kord*tord*ford*pord))
        for ki in range(kord):
            for ti in range(tord):
                for fi in range(ford):
                    fp.write("   ")
                    for pi in range(pord):
                        cf = coefs[ki, ti, fi, pi]
                        fp.write(" std::complex<double>(")
                        fp.write("{}, {})".format(cf.real, cf.imag))
                        if ki + 1 < kord:
                            fp.write(",")
                    fp.write("\n")
        fp.write("};\n")
        # Add frequency channels (not part of original format)
        fp.write("const double {}_channels[{}] = {{\n    ".format(
                 varprefix, len(artsdata['channels'])))
        fp.write("{}".format(",\n    ".join(
            [str(frq) for frq in artsdata['channels']])))
        fp.write("\n};\n")
    return filename


def convDPE2LOFARcc(antpat, freq_center, freq_range, HAcoefband=None,
                    HAcoefversion="def0", kord=2, tord=5, ford=5,
                    channels=None):
    """Convert a DualPolElem (or TVecFields or RadFarField) to a Hamaker-Arts
    LOFAR .cc file."""
    if channels is None:
        if isinstance(antpat, tvecfun.TVecFields):
            channels = antpat.getRs()
        elif isinstance(antpat, radfarfield.RadFarField) \
                or isinstance(antpat, dualpolelem.DualPolElem):
            channels = antpat.getfreqs()
    coefs = hamaker_coefs(antpat, freq_center, freq_range, kord=kord,
                          tord=tord, ford=ford)
    HAcoefnrelem = coefs.size
    artsdata = {'coefs': coefs, 'HAcoefversion': HAcoefversion,
                'HAcoefband': HAcoefband, 'HAcoefnrelem': HAcoefnrelem,
                'freq_center': freq_center, 'freq_range': freq_range,
                'channels': channels}
    filename = _write_LOFAR_HAcc(artsdata)
    return artsdata, filename


def _read_LOFAR_HAcc(coefsccfilename):
    """Read Hamaker-Arts coefficients from c++ header files used in the
    "lofar_element_response" code developed at ASTRON for LOFAR.

    These header files contains LOFAR specific constructs such as reference
    to "lba" and "hba", so it is not suitable for other projects.
    """
    NR_POLS = 2
    re_fcenter = r'[lh]ba_freq_center\s*=\s*(?P<centerstr>.*);'
    re_frange = r'[lh]ba_freq_range\s*=\s*(?P<rangestr>.*);'
    re_shape = r'[lh]ba_coeff_shape\[3\]\s*=\s*\{(?P<lstshp>[^\}]*)\};'
    re_hl_ba_coeffs_lst = \
        r'(?P<version>\w+)_(?P<band>[hl]ba)_coeff\s*\[\s*(?P<nrelem>\d+)\s*\]\s*=\s*\{(?P<cmplstr>[^\}]*)\}'
    re_cc_cmpl_coef = r'std::complex<double>\((.*?)\)'
    re_channels = r'[lh]ba_channels\[(?P<nrfrqs>\d+)\]\s*=\s*\{(?P<chnls>[^\}]*)\};'
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
    searchres = re.search(re_channels, coefsfile_content)
    if searchres:
        channels = [float(frq) for frq in
                    searchres.group('chnls').split(',')]
    else:
        channels = None
    # The coefficients are order now as follows:
    #   coefs[k,theta,freq,spherical-component].shape == (2,5,5,2)
    artsdata = {'coefs': coefs, 'HAcoefversion': HAcoefversion,
                'HAcoefband': HAcoefband, 'HAcoefnrelem': HAcoefnrelem,
                'freq_center': freq_center, 'freq_range': freq_range,
                'channels': channels}
    return artsdata


def convLOFARcc2DPE(inpfile, dpe_outfile=None):
    """Convert a LOFAR .cc file of a Hamaker-Arts model named inpfile to a
    a DualPolElem object.
    The channels argument specifies the nominal subband frequencies of the
    data.
    If dpe_outfile is given, a pickled instance is created with this name.
    """
    artsdata = _read_LOFAR_HAcc(inpfile)
    #artsdata['channels'] = channels
    HLBA = HamakerPolarimeter(artsdata)
    stnDPolel = dualpolelem.DualPolElem(HLBA)
    if dpe_outfile is not None:
        pickle.dump(stnDPolel, open(dpe_outfile, 'wb'), PICKLE_PROTO)
    return stnDPolel


def plotElemPat(artsdata, frequency=55.0e6):
    """Plots the HA antenna pattern over the entire Hemisphere."""
    THETA, PHI = pntsonsphere.ZenHemisphGrid()  # theta=0.2rad for zeni anomaly
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
    #artsdata = _read_LOFAR_HAcc('../../example_FF_files/DefaultCoeffHBA.cc')
    artsdata = _read_LOFAR_HAcc('../../../dreamBeam/dreambeam/telescopes/LOFAR/share/defaultCoeffHBA.cc')
    print(artsdata)
    exit()
    freq = 55e6
    SAMPFREQ = 100e6
    NR_CHANNELS = 512
    artsdata["channels"] = numpy.linspace(SAMPFREQ, 3*SAMPFREQ, 2*NR_CHANNELS, endpoint=False)
    _write_LOFAR_HAcc(artsdata)
    exit()
    LBAmod = HamakerPolarimeter(artsdata)
    freqarg = [freq]
    phiarg = [[0.1-5*math.pi/4]]
    thtarg = [[math.pi/2-1.1]]
    jones = LBAmod.getJonesAlong(freqarg, (thtarg, phiarg))
    # jones_1 = LBAmod._getJonesAlong_alt(freqarg, (thtarg, phiarg))
    print(jones)
    exit()
    plotElemPat(artsdata, freq)
