import numpy as np
from . import vsh
"""Hansen's Vector Spherical Harmonics. K1 is the TE modes while K2 is TM."""


# TobiaC (2015-10-20)
def Ksum(coefK, theta, phi):
    lstop = coefK.LMAX

    thetaphiSZorig = theta.shape
    theta = theta.flatten()
    phi = phi.flatten()
    x = np.cos(theta)
    sintheta = np.sin(theta)
    expiphi = np.exp(1j*phi)
    # ma=abs(m)
    thetaphiSZ = theta.shape

    nil = np.zeros(thetaphiSZ+(1,), dtype=complex)
    snil = np.squeeze(nil)
    Vecsum_1dt = snil
    Vecsum_1dp = snil

    for l in range(1, lstop+1):
        # For l, compute K1_theta, K1_phi for all m>=0 (m=0,1,...,l)
        if l == 1:
            # l==1, initialize recursion terms:
            K1_theta_Lm2pos = np.zeros(thetaphiSZ+(3,), dtype=complex)
            K1_theta_Lm2pos[:, 0] = snil
            K1_theta_Lm2pos[:, 1] = snil
            K1_theta_Lm2pos[:, 2] = snil

            K1_theta_Lm1pos = np.zeros(thetaphiSZ+(3,), dtype=complex)
            K1_theta_Lm1pos[:, 0] = snil
            K1_theta_Lm1pos[:, 1] = snil
            K1_theta_Lm1pos[:, 2] = snil

            K1_theta_Lm0pos = np.zeros((thetaphiSZ[0], 3), dtype=complex)
            K1_theta_Lm0pos[:, 1] = np.sqrt(3)/2.0*1j*np.exp(1j*phi)

            K1_phi_Lm0pos = np.zeros((thetaphiSZ[0], 3), dtype=complex)
        else:
            # m==1:l-1
            for m in range(1, (l-1)+1):
                K1_theta_Lm0pos[:, m] = \
                    -np.sqrt((l-1)*(2*l+1)/float((l-m)*(l+m)*(l+1)))*(
                    1j*np.sqrt((2*l-1))*x*K1_theta_Lm1pos[:, m]
                    -np.sqrt((l-2)*(l-m-1)*(l+m-1)/float((l)*(2*l-3)))
                    * K1_theta_Lm2pos[:, m])
                K1_phi_Lm0pos[:, m] = (-K1_theta_Lm1pos[:, m]
                      * np.sqrt((l-1)*(l-m)*(l+m)*(2*l+1)/float((l+1)*(2*l-1)))
                      +1j*K1_theta_Lm0pos[:, m]*x*l)/m
            # m==l
            K1_theta_Lm0pos[:, l] = (1j*K1_theta_Lm1pos[:, l-1]*sintheta
                * np.exp(1j*phi) * np.sqrt(l*(2*l+1)/float(2*(l-1)*(l+1))))
        # end if
        # m==0
        K1_theta_Lm0pos[:, 0] = snil
        K1_phi_Lm0pos[:, 0] = (1j*K1_theta_Lm0pos[:, 1]*np.exp(-1j*phi)
                               *sintheta*np.sqrt(l*(l+1)))
        # m==l
        # The K1_theta_Lm0pos[:,l] was already set above
        # (since formula not for l==1)
        K1_phi_Lm0pos[:, l] = 1j*K1_theta_Lm0pos[:, l]*x

        # m==l+1 as buffer
        # K1_theta_Lm0pos[:,l+1]=nil
        # K1_phi_Lm0pos[:,l+1]=nil
        K1_theta_Lm0pos = np.hstack((K1_theta_Lm0pos, nil))
        K1_phi_Lm0pos = np.hstack((K1_phi_Lm0pos, nil))

        # Shift down l-1, l to l'-2,l'-1
        K1_theta_Lm2pos = K1_theta_Lm1pos.copy()
        K1_theta_Lm1pos = K1_theta_Lm0pos.copy()

        # Set up m<0 masks
        negmtermsigns = np.power((-1), range(1, (l+1)))
        # Alternative: No (-1)^m factor for m<0
        # negmtermsigns=-1*np.ones(negmtermsigns.shape)

        # Sum up
        s = 1-1
        #    m>=0
        # Increment Vecsum with K1_theta, K1_phi for all m>=0
        Vecsum_1dt = Vecsum_1dt+np.dot(K1_theta_Lm0pos[:, 0:(l+1)],
                     np.array(coefK.coefs_snm[s][l-1])[0:(l+1)])
        Vecsum_1dp = Vecsum_1dp+np.dot(K1_phi_Lm0pos[:,0:(l+1)],
                     np.array(coefK.coefs_snm[s][l-1])[0:(l+1)])
        #    m<0
        # Increment Vecsum with K1_theta, K1_phi for all m<0 computed from m>0
        Lpowfac = (-1)**(l+1)
        # Lpowfac=1
        Vecsum_1dt = Vecsum_1dt+Lpowfac*np.dot(
                np.conj(K1_theta_Lm0pos[:, 1:(l+1)])*negmtermsigns,
                np.flipud(np.array(coefK.coefs_snm[s][l-1]))[0:(l)])
        Vecsum_1dp = Vecsum_1dp+Lpowfac*np.dot(
                np.conj(K1_phi_Lm0pos[:,1:(l+1)])*negmtermsigns,
                np.flipud(np.array(coefK.coefs_snm[s][l-1]))[0:(l)])
        s = 2-1
        #    m>=0
        # Increment Vecsum with K2_theta, K2_phi for all m>=0
        # computed from K1_theta, K1_phi with m>=0.
        Vecsum_1dt = Vecsum_1dt-1j*np.dot(K1_phi_Lm0pos[:, 0:(l+1)],
                         np.array(coefK.coefs_snm[s][l-1])[0:(l+1)])
        Vecsum_1dp = Vecsum_1dp+1j*np.dot(K1_theta_Lm0pos[:,0:(l+1)],
                         np.array(coefK.coefs_snm[s][l-1])[0:(l+1)])
        #    m<0
        # Increment Vecsum with K2_theta, K2_phi for all m<0
        # computed from K1_theta, K1_phi with m>=0.
        Vecsum_1dt = Vecsum_1dt-1j*Lpowfac*np.dot(
                        np.conj(K1_phi_Lm0pos[:, 1:(l+1)])*negmtermsigns,
                        np.flipud(np.array(coefK.coefs_snm[s][l-1]))[0:(l)])
        Vecsum_1dp = Vecsum_1dp+1j*Lpowfac*np.dot(
                        np.conj(K1_theta_Lm0pos[:, 1:(l+1)])*negmtermsigns,
                        np.flipud(np.array(coefK.coefs_snm[s][l-1]))[0:(l)])
        # End for l loop
    # Normalize:
    # 4*pi normalization:
    # vshnrm=4.0*np.pi
    # Unnormalized:
    vshnrm = 1.0
    Vecsum_1dt = vshnrm*Vecsum_1dt
    Vecsum_1dp = vshnrm*Vecsum_1dp

    # Reshape theta,phi dependence to original shape from 1D vec shape.
    Vecsum_t = Vecsum_1dt.reshape(thetaphiSZorig)
    Vecsum_p = Vecsum_1dp.reshape(thetaphiSZorig)
    return Vecsum_t, Vecsum_p
