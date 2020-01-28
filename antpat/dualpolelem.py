"""Model generic, a dual polarized antenna element."""
# TobiaC 2015-12-02

import numpy
import matplotlib.pyplot as plt
import matplotlib.dates
from antpat.reps.sphgridfun import pntsonsphere, tvecfun
from antpat.io.feko_ffe import FEKOffe
from antpat.radfarfield import RadFarField


class DualPolElem(object):
    """Main class for a dual-pol antenna element. It can be constructed
    from two generic representations: two radiation far-fields (two
    single pol antennas) or a tangential (Jones) matrix.
    """
    def __init__(self, *args):
        if len(args) == 0:
            self.tmfd   = None
        elif len(args) == 1:
            #"tmfd" stands for tangential matrix field on directions
            self.tmfd   = args[0]
            self.radFFp = None
            self.radFFq = None
        elif len(args) == 2:
            self.tmfd   = None
            self.radFFp = args[0]
            self.radFFq = args[1]
        else:
            raise RuntimeError("Not more than two arguments")
        self.basis = None

    def getfreqs(self):
        """Get Frequencies"""
        if self.tmfd is None:
            return self.radFFp.getfreqs()
        else:
            return self.tmfd.getfreqs()

    def getJonesPat(self,freqval):
        """Return the dual-pol antenna elements Jones pattern for a
        given frequency."""
        THETA, PHI, p_E_th, p_E_ph=self.radFFp.getFFongrid(freqval)
        THETA, PHI, q_E_th, q_E_ph=self.radFFq.getFFongrid(freqval)

        Jones=numpy.zeros(p_E_th.shape+(2,2), dtype=complex)
        Jones[...,0,0]=p_E_th
        Jones[...,0,1]=p_E_ph
        Jones[...,1,0]=q_E_th
        Jones[...,1,1]=q_E_ph
        return THETA, PHI, Jones

    def getJonesAlong(self, freqval, theta_phi_view):
        theta_view, phi_view = theta_phi_view
        (theta_build, phi_build) = self.view2build_coords(theta_view, phi_view)
        if self.tmfd is None:
            p_E_th, p_E_ph = self.radFFp.getFFalong_build(freqval,
                                              (theta_build, phi_build) )
            q_E_th, q_E_ph = self.radFFq.getFFalong_build(freqval,
                                              (theta_build, phi_build) )
            Jones=numpy.zeros(p_E_th.shape+(2,2), dtype=complex)
            Jones[...,0,0] = p_E_th
            Jones[...,0,1] = p_E_ph
            Jones[...,1,0] = q_E_th
            Jones[...,1,1] = q_E_ph
        else:
            Jones = self.tmfd.getJonesAlong(freqval,
                                              (theta_build, phi_build) )
            if self.basis is not None:
                p_E_th = Jones[...,0,0]
                p_E_ph = Jones[...,0,1]
                q_E_th = Jones[...,1,0]
                q_E_ph = Jones[...,1,1]
                p_E_th, p_E_ph=tvecfun.transfVecField2RotBasis(self.basis,
                                            (theta_build, phi_build),
                                            (p_E_th, p_E_ph))
                q_E_th, q_E_ph=tvecfun.transfVecField2RotBasis(self.basis,
                                            (theta_build, phi_build),
                                            (q_E_th, q_E_ph))
                Jones[...,0,0] = p_E_th
                Jones[...,0,1] = p_E_ph
                Jones[...,1,0] = q_E_th
                Jones[...,1,1] = q_E_ph
        return Jones

    def getFFalong(self, freqval,  theta_phi_view, polchan=0):
        jones = self.getJonesAlong(freqval,  theta_phi_view)
        E_th = jones[..., polchan, 0].squeeze()
        E_ph = jones[..., polchan, 1].squeeze()
        return E_th, E_ph

    def view2build_coords(self, theta_view, phi_view):
        """Get the corresponding directions in the build frame."""
        if self.basis is not None:
            (theta_build, phi_build) = pntsonsphere.rotToFrame(
                                           numpy.transpose(self.basis),
                                           theta_view, phi_view)
        else:
            (theta_build, phi_build) = (theta_view, phi_view)
        return (theta_build, phi_build)

    def rotateframe(self, rotMat):
        """Rotate the frame of antenna. This is a 'passive' rotation: it
        does not rotate the field, but when evaluated in some
        direction the direction given will be rotated to the frame so
        as to appear as if it were rotated.

        The basis or rotation matrix is to be considered as acting on
        the antenna, i.e.

              view_crds=rotMat*build_crds

        assuming the the antenna has not been rotated already. If it has then
        the inputted rotation is added to the current rotation, so that

              view_crds=rotMat*rotMat_0*build_crds

        where rotMat_0 is previous rotation state (could be aggregate of many).
        """
        if self.basis is None:
            self.basis = rotMat
        else:
            self.basis = numpy.matmul(rotMat, self.basis)

    def load_ffes(self, filename_p, filename_q):
        """Load a pair of FFE and make them correspond to this DualPolElem
        object. First file will be pol-channel p and second q."""
        ffefile_p = FEKOffe(filename_p)
        tvf_p = tvecfun.TVecFields()
        tvf_q = tvecfun.TVecFields()
        tvf_p.load_ffe(filename_p)
        tvf_q.load_ffe(filename_q)
        self.radFFp = RadFarField(tvf_p)
        self.radFFq = RadFarField(tvf_q)

    def load_ffe(self, filename, request_p=None, request_q=None):
        #FIX: This not the most efficient way to do this as it does two passes over feko file.
        ffefile = FEKOffe(filename)
        if request_p is None and request_q is None :
            if len(ffefile.Requests) == 2:
                requests = list(ffefile.Requests)
                requests.sort()  # # FIXME: Not sure how to order requests
                request_p = requests[0]
                request_q = requests[1]
            else:
                raise RuntimeError(
                    "File contains multiple FFs (specify one): "
                    + ','.join(ffefile.Requests))
        print("Request_p= "+request_p)
        print("Request_q= "+request_q)
        tvf_p = tvecfun.TVecFields()
        tvf_q = tvecfun.TVecFields()
        tvf_p.load_ffe(filename, request_p)
        tvf_q.load_ffe(filename, request_q)
        self.radFFp = RadFarField(tvf_p)
        self.radFFq = RadFarField(tvf_q)

    def plotJonesPat3D(self, freq=0.0, vcoord='sph',
                       projection='equirectangular', cmplx_rep='AbsAng'):
        """Plot the Jones pattern as two single pol antenna patterns."""
        theta_rad, phi_rad, JonesPat=self.getJonesPat(freq)
        Ep = numpy.squeeze(JonesPat[...,0,:])
        Eq = numpy.squeeze(JonesPat[...,1,:])
        tvecfun.plotvfonsph(theta_rad, phi_rad, numpy.squeeze(Ep[...,0]),
                            numpy.squeeze(Ep[...,1]), freq, vcoord,
                            projection, cmplx_rep, vfname='p-chan:'+self.radFFp.name)
        tvecfun.plotvfonsph(theta_rad, phi_rad, numpy.squeeze(Eq[...,0]),
                            numpy.squeeze(Eq[...,1]), freq, vcoord,
                            projection, cmplx_rep, vfname='q-chan:'+self.radFFp.name)


def plot_polcomp_dynspec(tims, frqs, jones):
    """Plot dynamic power spectra of each polarization component."""
    #fig, (ax0, ax1) = plt.subplots(nrows=2)
    p_ch = numpy.abs(jones[:,:,0,0].squeeze())**2+numpy.abs(jones[:,:,0,1].squeeze())**2
    q_ch = numpy.abs(jones[:,:,1,1].squeeze())**2+numpy.abs(jones[:,:,1,0].squeeze())**2
    ftims=matplotlib.dates.date2num(tims)
    dynspecunit = 'flux arb.'
    # In dB
    dBunit = False
    if dBunit:
        p_ch = 10*numpy.log10(p_ch)
        q_ch = 10*numpy.log10(q_ch)
        dynspecunit += ' dB'
    dynspecunit += ' unit'
    plt.figure()
    plt.subplot(211)
    plt.pcolormesh(numpy.asarray(tims), frqs, p_ch)
    plt.title('p-channel')
    #plt.clim(0, 1.0)
    plt.colorbar().set_label(dynspecunit)
    plt.subplot(212)
    plt.pcolormesh(numpy.asarray(tims), frqs, q_ch)
    plt.title('q-channel')
    #plt.clim(0, 1.0)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar().set_label(dynspecunit)
    plt.show()


def jones2gIXR(jones):
    U,s,V=numpy.linalg.svd(jones)
    g = (s[...,0]+s[...,1])*0.5
    cnd=s[...,0]/s[...,1]
    IXRJ=((1+cnd)/(1-cnd))**2
    return g, IXRJ


def ampgain2intensitygain(g):
    pass

def IXRJ2IXRM(IXRJ):
    """Convert Jones IXR to Mueller IXR. See Carozzi2011."""
    return (1+IXRJ)/(1*numpy.sqrt(IXRJ))
