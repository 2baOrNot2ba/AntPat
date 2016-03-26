"""
feko_ffe handles the FEKO file format for far-field patterns.
"""
#Originally written by Griffin Foster griffin.foster@gmail.com 

import numpy as np


class FEKOffeRequest(object):
    """Represents a far-field computed by FEKO. The far-field is one
    possible output 'request' in FEKO. It is an array of complex
    2-vectors on a regular grid in spherical polar coordinates."""
    def __init__(self, Name = None):
        self.etheta = []
        self.ephi   = []
        self.gtheta = []
        self.gphi   = []
        self.gtotal = []
        self.freqs  = []
        self.Name  = Name
    
    def _add_head(self, freqs, coord, stheta, sphi, rtype):
        self.freqs.append(freqs)
        self.coord = coord
        self.stheta = stheta
        self.sphi = sphi
        self.rtype = rtype
    
    def _add_data(self, theta, phi, etheta, ephi, gtheta, gphi, gtotal):
        self.theta = theta
        self.phi = phi
        self.etheta.append(etheta)
        self.ephi.append(ephi)
        self.gtheta.append(gtheta)
        self.gphi.append(gphi)
        self.gtotal.append(gtotal)


class FEKOffe(object):
    """A FEKOffe object holds the data contained in a FEKO .ffe file.
    This consists of one or more FEKOffeRequest ."""
    def __init__(self, fn, ftype='ascii'):
        self.comments = []
        self.Requests = set()
        self.Request = {}
        if ftype.startswith('ascii'):
           self.read(fn)
    
    def read(self, ffefile):
        """Reads in .ffe files."""
        fh=open(ffefile,'r')
        d=fh.read()
        fh.close()
        fieldsStr=d.split('#Request')
        lines=fieldsStr[0].split('\n')
        #parse main header (fieldsStr[0])
        for l in lines:
            if l=='': continue 
            elif l.startswith('**'):
                self.comments.append(l[2:])
                continue
            elif l.startswith('#'):
                #header information
                hdr=l.split('#')[-1]
                if hdr.lower().startswith('file type'): self.ftype=hdr.split(':')[-1].lstrip()
                elif hdr.lower().startswith('file format'): self.fformat=hdr.split(':')[-1].lstrip()
                elif hdr.lower().startswith('source'): self.source=hdr.split(':')[-1].lstrip()
                elif hdr.lower().startswith('date'): self.date=hdr.split('e:')[-1].lstrip()
        for field in fieldsStr[1:]:
            blocktxt='#Request'+field
            #I repeatedly read the theta & phi values but they're intentionally
            #overwritten every new request block, so only the last set is saved.
            lines=blocktxt.split('\n')
            theta=[]
            phi=[]
            etheta=[]
            ephi=[]
            gtheta=[]
            gphi=[]
            gtotal=[]
            for l in lines:
                if l=='' or l.startswith('*'): continue
                if l.startswith('#'):
                    #header information
                    hdr=l.split('#')[-1]
                    kv_val=hdr.split(':', 1)[-1].lstrip()
                    if hdr.lower().startswith('request'):
                        Request = kv_val
                    elif hdr.lower().startswith('freq'):
                        freqs = float(kv_val)
                    elif hdr.lower().startswith('coord'):
                        coord = kv_val
                    elif hdr.lower().startswith('no. of theta'):
                        stheta=int(kv_val)
                    elif hdr.lower().startswith('no. of phi'):
                        sphi=int(kv_val)
                    elif hdr.lower().startswith('result'):
                        rtype=kv_val
                    continue

                #assume data format:
                #"Theta" "Phi" "Re(Etheta)" "Im(Etheta)" "Re(Ephi)" "Im(Ephi)" "Gain(Theta)" "Gain(Phi)" "Gain(Total)"
                cleanStr=" ".join(l.split())+" "
                dlist=map(float,cleanStr.split(' ')[:-1])
                theta.append(dlist[0])
                phi.append(dlist[1])
                etheta.append(dlist[2]+1j*dlist[3])
                ephi.append(dlist[4]+1j*dlist[5])
                gtheta.append(dlist[6])
                gphi.append(dlist[7])
                gtotal.append(dlist[8])
            #Process block
            if Request not in self.Requests :
                self.Requests.add(Request)
                self.Request[Request] = FEKOffeRequest(Request)
            self.Request[Request]._add_head(freqs, coord, stheta, sphi, rtype)
            #Fixup the data variables
            theta =  np.array(theta ).reshape(stheta, sphi, order='F')
            phi =    np.array(phi   ).reshape(stheta, sphi, order='F')
            etheta = np.array(etheta).reshape(stheta, sphi, order='F')
            ephi =   np.array(ephi  ).reshape(stheta, sphi, order='F')
            gtheta = np.array(gtheta).reshape(stheta, sphi, order='F')
            gphi =   np.array(gphi  ).reshape(stheta, sphi, order='F')
            gtotal = np.array(gtotal).reshape(stheta, sphi, order='F')
            self.Request[Request]._add_data(theta, phi, etheta, ephi, gtheta, gphi, gtotal)
    
    def write(self, fn):
        """Write data to ASCII .ffe file"""
        fh=open(fn,'w')
        #main header
        ostr=''
        ostr+='##File Type: '+self.ftype+'\n'
        ostr+='##File Format: '+self.fformat+'\n'
        ostr+='##Source: '+self.source+'\n'
        ostr+='##Date: '+self.date+'\n'
        ostr+='**'+'\n'.join(self.comments)+'\n'
        ostr+='** '+'File exported by feko_ff.py'+'\n'
        for req in self.Requests:
            f = self.Request[req]
            line_end_pad = '   \n'
            data_format = '               '+('   % 13.8E'*9)+line_end_pad
            for freqind, freq in enumerate(f.freqs):
                #write header and data
                ostr+='#Request Name: '+f.Name+'\n'
                ostr+='#Frequency: '+str(f.freqs[freqind])+'\n'
                ostr+='#Coordinate System: '+f.coord+'\n'
                ostr+='#No. of Theta Samples: '+str(f.stheta)+'\n'
                ostr+='#No. of Phi Samples: '+str(f.sphi)+'\n'
                ostr+='#Result Type: '+f.rtype+'\n'
                ostr+='#No. of Header Lines: 1\n'
                ostr+=(
'#                 \"Theta\"           \"Phi\"             \"Re(Etheta)\"      \"Im(Etheta)\"      \"Re(Ephi)\"        \"Im(Ephi)\"        \"Gain(Theta)\"     \"Gain(Phi)\"       \"Gain(Total)\"  '+line_end_pad)
                etheta = f.etheta[freqind]
                ephi = f.ephi[freqind]
                gtheta = f.gtheta[freqind]
                gphi = f.gphi[freqind]
                gtotal = f.gtotal[freqind]
                for pind in range(f.sphi):
                    for tind in range(f.stheta):
                        ostr+=data_format%(
                        f.theta[tind,pind], f.phi[tind,pind],
                        etheta[tind,pind].real, etheta[tind,pind].imag,
                        ephi[tind,pind].real, ephi[tind, pind].imag,
                        gtheta[tind,pind], gphi[tind,pind],
                        gtotal[tind,pind])
        fh.write(ostr)
        fh.close()


if __name__ == '__main__':
    print 'Running test cases...'
    print 'Reading FEKO file'
    fekoFile = FEKOffe('../../example_FF_files/FFE/PAPER_FF_X.ffe')
    filecopyname = 'test_feko.ffe'
    for req in fekoFile.Requests:
        print req
        print fekoFile.Request[req].freqs
        frqch = 0
        print "For freqchan:"+str(frqch)+" the E_phi field e.g. is\n"
        print fekoFile.Request[req].ephi[frqch]
    print 'Writing FEKO data to file'
    fekoFile.write(filecopyname)
    print 'Read back that text file to make sure it is correctly formatted'
    fekoFileredux=FEKOffe(filecopyname)
    print '...Made it through without errors'

