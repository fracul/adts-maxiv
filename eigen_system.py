import numpy as np
import math
from scipy import constants, special, optimize

const_cg = 8.85e-32

class ImpedanceFromFile:

    def __init__(self,filename):
        """
        filename with column order: freq [Hz], Re(Z), Im(Z) [Ohm], [Ohm/m]
        """
        self.imp = np.loadtxt(filename,unpack=True)
        self.freq = self.imp[0]
        self.impedance = self.imp[1]+1j*self.imp[2]

    def calcImpedance(self,frequency):
        
        return np.interp(frequency,self.freq,self.impedance)

class ImpedanceModel:

    def __init__(self,rhorw=0,radius=0.011,length=528.,
                 rs=0,rfreq=5000,qfact=1,
                 coat_thickness=0,coat_rhorw=0,round_pipe=False,filename=''):
        """
        Create a transverse impedance model.
        Arguments:

        *rhorw* - resistivity of vacuum chamber [Ohms m], default: 0 
        *radius* - half-aperture of vacuum chamber [m], default: 0.011 
        *length* - length of the vacuum chamber (ring) [m], default: 528
        *rs* - shunt impedance of broadband resonator impedance [Ohms/m], default: 0
        *rfreq* - resonant frequency of broadband resonator impedance [Hz], default: 5000
        *qfact* - quality factor of broadband resonator impedance [], default 1
        *coat_thickness* - thickness of vacuum chamber coating [m], default: 0
        *coat_rhorw* - resistivity of vacuum chamber coating [Ohms m], default: 0
        *round_pipe* - impedance model for round vacuum chamber (only affects coated chamber model), default: False
        *filename* - arbitrary impedance read from file
        """

        self.rhorw = rhorw
        self.radius = radius
        self.length = length
        self.revfreq = constants.c/self.length
        self.rwconst = constants.mu_0*self.length/(2*np.pi*self.radius**3)*constants.c
        self.rwskinconst = self.rhorw/(np.pi*constants.mu_0)
        self.rs = rs
        self.qfact = qfact
        self.rfreq = rfreq
        self.bw = self.rfreq/self.qfact

        self.rwconst_coat = -1j*np.sqrt(constants.mu_0/constants.epsilon_0)/2./np.pi*self.length
        self.rhorw_coat = coat_rhorw
        self.rwskinconst_coat = self.rhorw_coat/(np.pi*constants.mu_0)
        self.thick_coat = coat_thickness
        self.round_pipe = round_pipe

        self.fromfile = False
        if filename:
            self.fromfile = True
            self.imp = np.loadtxt(filename,unpack=True)
            numfreq = len(self.imp[0])
            self.freq = np.zeros(numfreq*2)
            self.impedance = np.zeros(numfreq*2,complex)
            self.freq[:numfreq] = -self.imp[0][numfreq::-1]*1e9
            self.freq[numfreq:] = self.imp[0]*1e9
            self.impedance[numfreq:] = -(self.imp[1]+1j*self.imp[2])*1e3
            self.impedance[:numfreq] = 1e3*(self.imp[1]-1j*self.imp[2])[numfreq::-1]

    def zSigmaY(self,frequency):

        kdiff = 1/self.radius/1e3
        kran = np.arange(1,7000,1)*kdiff
        if isinstance(frequency,np.ndarray):
            tmplt = np.ones((len(frequency),len(kran)))
            kran = tmplt*kran
            frequency = (tmplt.T*frequency).T

        q0 = kran
        
        qreal = -(2*np.pi*frequency)**2*constants.mu_0/constants.c**2
        q1 = np.sqrt(qreal+2*1j/self.rwskinconst_coat/np.pi*frequency+kran**2)
        q2 = np.sqrt(qreal+2*1j/self.rwskinconst/np.pi*frequency+kran**2)

        q21 = q2/q1
        q10 = q1/q0
        ka = kran*self.radius

        c0 = np.cosh(ka)
        s0 = np.sinh(ka)
        t1 = np.tanh(q1*self.thick_coat)
        #if (t1<1e-300).any():
        #    print 'UNDER'
        #    t1[t1<1e-300] = 1
        exp0 = np.exp(kran*self.radius)

        oneminusgy = (1+q21*t1)/(c0+q21*q10*np.sinh(s0)+(q21*c0+q10*s0)*t1)

        return self.rwconst_coat*np.trapz(kran/s0*oneminusgy,dx=kdiff)

    def zSigmaYRound(self,frequency):

        kppa1 = np.sqrt(-1j*2*np.pi*frequency*constants.mu_0/self.rhorw_coat)

        shb = np.sinh(kppa1*self.thick_coat)
        chb = np.cosh(kppa1*self.thick_coat)
        shbp = np.cosh(kppa1*self.thick_coat)
        chbp = np.sinh(kppa1*self.thick_coat)

        #kppa2 = np.sqrt(-1j*8*np.pi**2*frequency/self.rhorw/constants.c**2)
        kppa2 = np.sqrt(-1j*2*np.pi*frequency*constants.mu_0/self.rhorw)
        kppa21 = kppa2/kppa1
        #1/0

        imped = self.rwconst_coat*2/self.radius/self.radius*(shbp+kppa21*shb)/(shbp+kppa21*kppa1*self.radius*chb+kppa21*shb+kppa1*self.radius*chbp)
        #imped.real *= -1
        imped = -imped.conj()

        return imped

    def calcImpedance(self,frequency):
        """
        Calculate impedance at *frequency* [Hz], float or numpy array of floats.
        """

        fr = frequency
        if self.rwskinconst_coat!=0: 
            if self.round_pipe: imp_resist = -self.zSigmaYRound(frequency).conj()
            else: imp_resist = -self.zSigmaY(frequency).conj()
        elif self.rwskinconst!=0: imp_resist = self.rwconst*(np.sign(fr)-1j)*np.sqrt(self.rwskinconst/np.absolute(fr))
        else: imp_resist = 0

        ones_fr = fr/fr
        rs = np.outer(self.rs,ones_fr)
        rfreq = np.outer(self.rfreq,ones_fr)
        bw = np.outer(self.bw,ones_fr)
        imp_resonant = np.sum(rs*rfreq/(fr+1j/bw*(rfreq**2-fr**2)),axis=0)

        imp_file = 0
        if self.fromfile:
            imp_file = np.interp(frequency,self.freq,self.impedance.real)+1j*np.interp(frequency,self.freq,self.impedance.imag)

        return imp_resist+imp_resonant+imp_file

class MultiSystem:

    def __init__(self,sring,impmodel,truncate,nazimuth,
                 multibunch=True,lag_basis=False,fokk_planck=False,exact=False,trans_tau=False):

        self.impmodel = impmodel
        self.revfreq = sring.frf/sring.nbunch
        self.truncate = truncate
        self.blen = sring.blen
        self.sring = sring
        self.multi = multibunch
        self.azimuth = nazimuth
        self.mu = self.sring.nbunch-1
        self.revfreq = self.sring.frf/self.sring.nbunch
        self.lag_basis = lag_basis
        self.fokk_planck = fokk_planck
        self.trans_tau = trans_tau
        self.nmodes = 2*truncate+1
        self.tauz = 2*self.sring.length*self.sring.rho0/(const_cg*constants.c*self.sring.energy**3)
        self.taue = self.tauz/self.sring.je
        self.exact = exact
        if self.lag_basis>0 and not self.trans_tau:
            self.nmodes = self.lag_basis

    def constructMatrix(self):

        self.const = self.sring.current*self.sring.betay*self.revfreq/self.sring.energy/2.
        self.matrix = np.zeros((self.truncate,self.truncate),complex)
        fractune = self.sring.vtune-int(self.sring.vtune)
        if self.multi:
            self.freq = (np.arange(-self.truncate,self.truncate+1)*self.sring.nbunch+self.mu
                         +fractune+self.azimuth*self.sring.ltune)*self.revfreq
        else:
            self.freq = (np.arange(-self.truncate,self.truncate+1)
                         +fractune+self.azimuth*self.sring.ltune)*self.revfreq
        self.freq = self.freq[self.freq!=0]
        self.chrofreq = self.sring.vchro/self.sring.alphac*self.revfreq
        self.impedance = self.impmodel.calcImpedance(self.freq)
        modefr = (self.freq-self.chrofreq)*2*np.pi
        expfreq = np.exp(-(modefr*self.blen)**2)
        self.matrix = special.iv(self.azimuth,np.outer(modefr,modefr)*self.blen**2)*np.sqrt(np.outer(expfreq,expfreq))
        self.matrix = 1j*self.matrix*self.impedance*self.const
        if self.lag_basis:
            self.basisMatrix(self.lag_basis)
            if self.trans_tau:
                self.matrix = self.matrix-1j*np.dot(self.basis_matrix,
                                                    np.dot(np.eye(self.lag_basis)*(2*self.azimuth+np.arange(self.lag_basis))/self.taue,
                                                           self.basis_matrix_inv))
            else:
                self.nmodes = self.lag_basis
                self.matrix = np.dot(self.basis_matrix_inv,np.dot(self.matrix,self.basis_matrix))
                if self.fokk_planck:
                    self.matrix = self.matrix-1j*np.eye(self.nmodes)*(2*self.azimuth+np.arange(self.nmodes))/self.taue

    def basisMatrix(self,ktrunc=0):

        import sympy

        if ktrunc<=0: ktrunc = 2*self.truncate+1
        t1 = (2*np.pi*(self.freq-self.chrofreq)*self.sring.blen)**2/2.
        tsize = 2*self.truncate+1
        tmplte = np.ones((tsize,ktrunc))
        t1mat = (tmplte.T*t1).T
        #t1matfact = tmplte*np.array([np.prod(np.arange(1,p,dtype=float)) for p in np.arange(ktrunc)])
        t1matfact = tmplte*np.array([math.factorial(p) for p in np.arange(ktrunc)])
        t1matfact_azimuth = tmplte*np.array([math.factorial(p+self.azimuth) for p in np.arange(ktrunc)])
        pows = tmplte*np.arange(ktrunc)
        self.basis_matrix = t1mat**(self.azimuth/2.+pows)*2.**(self.azimuth/2.)*np.exp(-t1mat)\
                            *np.prod(np.arange(1,self.azimuth))/np.sqrt(t1matfact*t1matfact_azimuth)#/self.blen**self.azimuth
        if self.exact and self.basis_matrix.shape[0]==self.basis_matrix.shape[1]:
            g = sympy.Matrix(self.basis_matrix)
            self.basis_matrix_inv = np.asarray(g.inv()).astype(float)
        else:
            self.basis_matrix_inv = np.linalg.pinv(self.basis_matrix,rcond=1e-5)

    def solvEigen(self):

        self.eigenfreqs, self.eigenmodes = np.linalg.eig(self.matrix)

    def getNumModes(self):
        return self.nmodes        

    def calcIth(self):

        self.grate = self.eigenfreqs.imag
        self.ith = self.sring.current/(self.eigenfreqs.imag*self.tauz)

class EigenSystem:
    """
    Class for calculating current-dependent tune shift and TMCI using Chin and Fokker-Planck formulae
    """

    def __init__(self,sring,impmodel,truncate,nazimuth,nradial,radimin=0,multibunch=False,plane='X',fokk_planck=False,dispmatsize=1,dispcoeff=0,chinint=False,dispint=False,precision128=False):
        """
        Class initialisation. Arguments:

        *sring* - utility.StorageRing instance (loaded from a file)
        *impmodel* - eigen_system.ImpedanceModel instance
        *truncate* - truncate summations at this number of revolution frequencies
        *nazimuth* - number of azimuthal head-tail modes to consider
        *nradial* - number of radial head-tail modes to consider
        
        Keyword arguments:

        *radimin* - minimum radial head-tail mode to consider, default: 0
        *multibunch* - calculate for a uniform machine fill instead of single bunch, default: False
        *plane* - 'X' for horizontal and 'Y' for vertical, default: 'X'
        *fokk_planck* - consider diffusion due to synchrotron radiation, default: False
        *dispmatsize* - size of dispersion matrix when including amplitude-dependent tune shift via Besnier's method. - DO NOT TRUST!
        *dispcoeff* - dispersion coefficient quantifying amplitude-dependent tune shift, default: 0
        *chinint* - use the Chin dispersion integral method instead of the dispersion matrix method  - DO NOT TRUST!
        *dispint* - use the Metral dispersion integral (to see if the matrix determinant evaluates to zero) - USE THIS ONE

        After initialisaion, the *constructMatrix* and *solvEigen* member functions are typically called to obtain results (coherent tune shifts).
        """

        self.impmodel = impmodel
        self.revfreq = sring.frf/sring.nbunch
        self.truncate = truncate
        if isinstance(nazimuth,(list,np.ndarray,tuple)): [self.azimin,self.azimax] = nazimuth
        else: [self.azimin,self.azimax] = [-nazimuth,nazimuth+1]
        self.nradial = nradial
        self.radimin = radimin
        self.azicount = self.azimax-self.azimin
        self.nmodes = nradial*self.azicount
        self.blen = sring.blen
        self.blen_norm = sring.blen*constants.c/sring.length*2*np.pi
        self.charge = sring.current/self.revfreq
        self.sring = sring
        self.multi = multibunch
        self.fokk_planck = fokk_planck
        self.mu = -1
        self.tauz = 2*self.sring.length*self.sring.rho0/(const_cg*constants.c*self.sring.energy**3)
        self.taue = self.tauz/self.sring.je
        self.dispmatsize = dispmatsize
        self.chinint = chinint
        self.dispint = dispint
        self.adts = dispcoeff
        if self.adts==0: self.dispmatsize=1
        if precision128:
            self.dt = np.complex128
        else:
            self.dt = complex
        if plane=='X':
            self.beta = self.sring.betax
            self.tune = self.sring.htune
        elif plane=='Y':
            self.beta = self.sring.betay
            self.tune = self.sring.vtune

    def getNumModes(self):
        return self.nmodes

    def i_mk(self,m,k):
        
        eps_m = 1
        if m<0: eps_m = (-1)**m
        absm = math.fabs(m)
        
        return np.array(eps_m/np.sqrt(math.factorial(absm+k)*math.factorial(k))*(self.fract)**(absm+2*k)*np.exp(-self.fract**2),dtype=complex)
        
    def constructMatrix(self):

        #self.bigk = self.sring.betay*np.pi/(self.sring.ltune*self.sring.energy)*self.charge*self.revfreq
        self.bigk = self.beta/(4*np.pi*self.sring.ltune*self.sring.energy)*self.charge*self.revfreq
        self.matrix = np.zeros((self.nmodes*self.dispmatsize,self.nmodes*self.dispmatsize),dtype=self.dt)
        self.chrofreq = self.sring.vchro/self.sring.alphac        
        if self.multi: 
            self.freq = np.arange(-self.truncate,self.truncate+1)*self.sring.nbunch+self.mu-self.chrofreq+self.tune-int(self.tune)#MAKES NO DIFFERENCE
#            self.bigk *= self.sring.nbunch
        else:
            self.freq = np.arange(-self.truncate,self.truncate+1)-self.chrofreq+self.tune-int(self.tune)
        self.fract = self.freq*self.blen_norm/np.sqrt(2)
#        if not hasattr(self,'impedance'): self.impedance = self.impmodel.calcImpedance((self.freq+self.chrofreq)*self.revfreq)#+self.sring.vchro/self.sring.alphac)MAKES NO DIFFERENCE
        self.impedance = self.impmodel.calcImpedance((self.freq+self.chrofreq)*self.revfreq)#+self.sring.vchro/self.sring.alphac)MAKES NO DIFFERENCE
        self.azi_index = np.zeros(self.nmodes)
        self.rad_index = np.zeros(self.nmodes)
        if self.dispint:
            ksi = np.arange(-50,50,0.1)
            self.dispints = np.array([1/dispersionIntegral(ksi+b*2*np.pi*self.revfreq*self.sring.ltune/2/self.dispint,self.dispint)/(2*np.pi*self.revfreq)/self.sring.ltune for b in range(self.azimin,self.azimax)])
            self.matrix = np.zeros(self.matrix.shape+(len(self.dispints[0]),),complex)
        for r in xrange(self.radimin,self.radimin+self.nradial):
            for a in xrange(self.azimin,self.azimax):
                imk = self.i_mk(a,r)
                dimone = (r-self.radimin)*self.azicount+a-0*self.radimin-self.azimin
                self.azi_index[dimone] = a
                self.rad_index[dimone] = r
                for s in xrange(self.radimin,self.radimin+self.nradial):
                    for b in xrange(self.azimin,self.azimax):
                        ikl = self.i_mk(b,s)
                        dimtwo = (s-self.radimin)*self.azicount+b-0*self.radimin-self.azimin
                        self.matrix[dimone,dimtwo] = 1j**(a-b-1)*self.bigk*np.sum(self.impedance*imk*ikl)+(dimone==dimtwo)*a
                        if dimone==dimtwo:
                            if self.fokk_planck:# and dimone==dimtwo:
                                absb = np.absolute(b)
                                diffuse = -1/2./self.taue*(np.sqrt(s*(s+absb+1))+np.sqrt((s+1)*(s+absb)))
                                diffuse = (absb+2*s)/self.taue
                                self.matrix[dimone,dimtwo] += 1j*diffuse/(2*np.pi*self.revfreq)/self.sring.ltune
                            if self.dispint:
                                self.matrix[dimone,dimtwo] += self.dispints[b-self.azimin]
        if self.adts!=0 and not self.chinint:
            tmplte = np.diag(np.ones(self.nmodes))*2*self.adts
            azidiag = np.diag(self.azi_index)
            for d in xrange(self.dispmatsize):
                self.matrix[d*self.nmodes:(d+1)*self.nmodes,d*self.nmodes:(d+1)*self.nmodes] -= tmplte*2*(d+1)-(d!=0)*azidiag
                if (d+1<self.dispmatsize):
                    self.matrix[(d+1)*self.nmodes:(d+2)*self.nmodes,d*self.nmodes:(d+1)*self.nmodes] += tmplte*np.sqrt((d+2)*(d+1))
                    self.matrix[d*self.nmodes:(d+1)*self.nmodes,(d+1)*self.nmodes:(d+2)*self.nmodes] += tmplte*np.sqrt((d+2)*(d+1))
        if self.dispint:
            self.matrix = self.matrix.transpose(2,0,1)

    def solvEigen(self):

        self.eigenfreqs, self.eigenmodes = np.linalg.eig(self.matrix)
        if self.chinint: self.dispChinAll()

    def spectrumMatrix(self,ptrunc=0,es_exact=False):

        import sympy

        if ptrunc<=0: ptrunc = self.nmodes/2
        freqmat = np.ones((2*ptrunc,self.nmodes))
        freqmat = (freqmat.T*np.arange(-ptrunc,ptrunc)).T
        self.freq = freqmat*self.sring.nbunch+self.mu+self.azi_index*self.sring.ltune+self.tune-int(self.tune)
        tp  = 2*np.pi*self.freq*self.revfreq*self.sring.blen/np.sqrt(2.)
        azi_abs = np.absolute(self.azi_index)
        self.spectmat = tp**(azi_abs+2*self.rad_index)*np.exp(-tp*tp)*2**(azi_abs/2.)*math.factorial(azi_abs)/np.sqrt(math.factorial(self.rad_index)*math.factorial(self.rad_index+azi_abs))
        #self.spectmat = self.spectmat.T
        #if es_exact and self.spectmat.shape[0]==self.spectmat.shape[1]:
        #    g = sympy.Matrix(self.spectmat)
        #    self.spectmat_inv = np.asarray(g.inv()).astype(float)
        #else:
        #    self.spectmat_inv = np.linalg.pinv(self.spectmat,rcond=1e-5)

    def estimateTuneShift(self,use_beta=True):

        if use_beta: self.const = -8*np.pi**(1.5)*self.sring.energy*self.sring.blen/self.sring.length*constants.c/self.beta
        else: self.const = -16*np.pi**(2.5)*self.sring.energy*self.sring.blen/self.sring.length**2*constants.c*self.tune
        #self.const = -16*np.pi**(2.5)*self.sring.energy*self.sring.blen/self.sring.length*constants.c/self.beta#self.sring.length**2*constants.c**2
        self.freq = np.arange(-self.truncate,self.truncate+1)-self.sring.vchro/self.sring.alphac-self.tune+int(self.tune)
        self.impedance = self.impmodel.calcImpedance(self.freq*self.revfreq)
        bspect = np.exp(-(2*np.pi*self.freq*self.revfreq*self.sring.blen)**2)
        self.zeff = np.trapz(bspect*self.impedance)/np.trapz(bspect)

        return self.zeff/self.const

    def dispMatrixChin(self,fguess):

        def detmatrix(tune_coherent):
            tau = (tune_coherent[0]+1j*tune_coherent[1]-self.azi_index*self.sring.ltune-self.tune)/2./self.adts/self.sring.ltune
            fnu_m = 1/2./self.adts*(1-tau*np.exp(tau)*special.expi(tau))
            matdet = np.linalg.det(np.diag(1/fnu_m)-self.matrix)
            return np.array([np.real(matdet),np.imag(matdet)])#*matdet.conj()

        #res = optimize.fmin(detmatrix,np.array([fguess.real,fguess.imag]),full_output=True,disp=False)
        res = optimize.root(detmatrix,np.array([fguess.real,fguess.imag]))#,full_output=True,disp=False)        

        return res

    def dispChinAll(self):

        self.eigenfreqs_noadts = self.eigenfreqs.copy()
        self.warnflag_chin = np.zeros(self.eigenfreqs.shape[0],int)
        self.magdet_chin = np.zeros(self.eigenfreqs.shape[0],complex)
        for i,e in enumerate(self.eigenfreqs_noadts):
            res = self.dispMatrixChin(e)
            #self.eigenfreqs[i] = res[0][0]+1j*res[0][1]
            self.eigenfreqs[i] = res.x[0]+1j*res.x[1]
            #self.magdet_chin[i] = res[1]            
            #self.warnflag_chin[i] = res[4]
        print('Eigenvalue solution ended with %d warnings' % np.sum(self.warnflag_chin))

class Scan:
    """
    Class for setting up scans to be able to scan variables such as bunch current and chromaticity
    """

    def __init__(self,esys,scan_vals,zipvals=[],breakthresh=None):
        """
        Class initialisation, arguments:
        
        *esys* - EigenSystem instance
        *scan_vals* - dictionary with scan-type names as key and scan values as the item. If greater than length 1,
                      scans will be nested.
                      Alternatives are:
                      > 'current' - beam current
                      > 'currblen' - beam current and bunch length according to effective longitudinal impedance
                                     listed for the storage ring
                      > 'adts' - amplitude-dependent tune shift
                      > 'qfact' - quality factor of broadband resonator impedance
                      > 'rfreq' - resonant frequency of broadband resonator impedance
                      > 'rs' - shunt impedance of broadband resonator impedance
                      > 'mu' - coupled-bunch mode number (multibunch only)

        Keyword arguments:
        *zipvals* - values that can be varied in parallel to the values defined in scan_vals, default: []
        *breakthresh* - growth rate at which instability is detected and calculation is cut short, default None

        After initialisaion, the *scan* member function typically called to obtain results (coherent tune shifts).        
        """
        
        self.scan_vals = scan_vals
        self.esys = esys
        self.nmodes = self.esys.getNumModes()
        self.zipvals = zipvals
        if breakthresh==None: self.breakthresh = {}
        else: self.breakthresh = breakthresh

    def scan(self,savemodes=False):
        """
        Start the scan. Single argument *savemodes* if eigen modes are to be returned as well as tune shift/growth rates (default: False).
        """

        kys = self.scan_vals.keys()
        for z in self.zipvals:
            kys.remove(self.zipvals[z])
            kys.append(self.zipvals[z])
        key = kys[0]
        #if key=='eloss' and len(kys)>1:
        #    key = kys[1]
        if key=='qfact' and len(self.scan_vals)>1: key = self.scan_vals.keys()[1]
        print key

        if key=='current': scanfunc = self._currentAdjust
        elif key=='mu': 
            self.esys.multi = True
            scanfunc = self._muAdjust
        elif key=='rs': scanfunc = self._rsAdjust
        elif key=='rfreq': scanfunc = self._rfreqAdjust
        elif key=='qfact': 
            self.qfact0 = 1*self.esys.impmodel.qfact
            self.rs0 = 1*self.esys.impmodel.rs
            scanfunc = self._qfactorAdjust
        elif key=='currblen': 
            self.blen0 = 1*self.esys.sring.blen
            scanfunc = self._currBlenAdjust
        elif key=='adts': scanfunc = self._adtsAdjust
        elif key=='eloss': scanfunc = self._u0Adjust
        else: scanfunc = self._chromaticityAdjust
        vals = self.scan_vals[key]

        if len(self.scan_vals)>1:
            results = []
            eigmodes = []
            scanv_tmp = self.scan_vals.copy()
            scanv_tmp.pop(key)

            for i,c in enumerate(vals):
                scanfunc(c)
                if key in self.zipvals:
                    scanv_tmp2 = scanv_tmp.copy()
                    scanv_tmp2.update({self.zipvals[key]:scanv_tmp[self.zipvals[key]][i]})
                    newscan = Scan(self.esys,scanv_tmp2,breakthresh=self.breakthresh)
                else: newscan = Scan(self.esys,scanv_tmp,breakthresh=self.breakthresh)
                newscan.scan(savemodes)
                results.append(newscan.results[:])
                if savemodes: eigmodes.append(newscan.eigenmodes[:])
            self.results = np.array(results,complex)
            if savemodes: self.eigenmodes = np.array(eigmodes,complex)
        else:
            breaknext = False
            self.results = np.zeros((len(vals),self.nmodes*self.esys.dispmatsize),complex)
            if savemodes: self.eigenmodes = np.zeros((len(vals),self.nmodes*self.esys.dispmatsize),complex)
        
            for i,c in enumerate(vals):
                print c
                scanfunc(c)
                try:
                    self.esys.constructMatrix()
                    self.esys.solvEigen()
                except np.linalg.linalg.LinAlgError:
                    pass
                else:
                    self.results[i] = self.esys.eigenfreqs[:]
                    if savemodes: self.eigenmodes[i] = self.esys.eigenmodes[:,np.argmin(self.esys.eigenfreqs.imag)]                    
                    if breaknext:
                        print 'Breaking at value', c                        
                        breaknext = False
                        break
                    if key in self.breakthresh and np.amin(self.esys.eigenfreqs.imag)<self.breakthresh[key]:
                        if i==0: breaknext = True
                        else:
                            print 'Breaking at value', c
                            break

    def localThreshold(self,fitstart=0,fitend=2):
        grate = np.amin(self.results.imag,axis=-1)
        if isinstance(self.esys,EigenSystem):
            grate *= self.esys.sring.ltune*self.esys.revfreq*2*np.pi
            
        if 'current' in self.breakthresh:
            inds = np.where((grate==0) & (self.scan_vals['current']!=0))
            gfit = np.zeros((2,self.results.shape[0]))
            for n in range(self.results.shape[0]):
                ithind = np.amin(inds[1][inds[0]==n])
                print ithind
                print self.scan_vals['current'][n,ithind-2:ithind], -self.esys.tauz*grate[n,ithind-2:ithind]
                gfit[:,n] = np.polyfit(self.scan_vals['current'][n,ithind-2:ithind],-self.esys.tauz*grate[n,ithind-2:ithind],1)
        elif self.zipvals:
            gfit = np.polyfit(self.scan_vals['current'][:,fitstart:fitend].T,-self.esys.tauz*grate[:,fitstart:fitend].T,1)            
        else: gfit = np.polyfit(self.scan_vals['current'][fitstart:fitend],-self.esys.tauz*grate[:,fitstart:fitend].T,1)
        
        self.ith = (1-gfit[1])/gfit[0]

    def calcIth(self):
        grate = np.amin(self.results.imag,axis=-1)
        if isinstance(self.esys,EigenSystem):
            grate *= self.esys.sring.ltune*self.esys.revfreq*2*np.pi
        self.ith = -self.esys.sring.current/grate/self.esys.tauz

    def _currentAdjust(self,c):
        self.esys.sring.current = c
        self.esys.charge = c/self.esys.revfreq

    def _chromaticityAdjust(self,c):
        self.esys.sring.vchro = c

    def _muAdjust(self,c):
        self.esys.mu = c

    def _rsAdjust(self,c):
        self.esys.impmodel.rs = c

    def _rfreqAdjust(self,c):
        self.esys.impmodel.rfreq = c

    def _qfactorAdjust(self,c):
        self.esys.impmodel.rs = self.rs0*c/self.qfact0
        self.esys.impmodel.qfact = c
        self.esys.impmodel.bw = self.esys.impmodel.rfreq/self.esys.impmodel.qfact

    def _adtsAdjust(self,c):
        self.esys.adts = c
        print self.esys.adts

    def _u0Adjust(self,c):
        self.esys.sring.eloss = c
        self.esys.sring.ltune = calcQso(self.esys.sring)
        self.esys.blen = self.esys.sring.alphac*self.esys.sring.espread/2./np.pi/self.esys.revfreq/self.esys.sring.ltune

    def _currBlenAdjust(self,c):

        self.esys.charge = c/self.esys.revfreq
        sr = self.esys.sring

        zotter = lambda ib: sr.alphac*ib/(sr.ltune*sr.energy)*(self.esys.sring.length/self.blen0/3e8)**3/np.pi*sr.zleff
        func = lambda x: np.absolute(x**3-x-zotter(c))

        self.esys.blen = optimize.fmin(func,sr.blen/self.blen0)*self.blen0
        self.esys.blen_norm = self.esys.blen*constants.c/self.esys.sring.length*2*np.pi

def readMosesOut(filename):

    f = open(filename,'r')
    valuedict = {}
    read_results = False
    current = []
    blen = []
    nus = []
    firstthru = True
    line = []
    for g in f:
        if '=' in g:
            sp = g.split()
            eqind = sp.index('=')
            try: valuedict.update({sp[eqind-1]:float(sp[eqind+1])})
            except ValueError: valuedict.update({sp[eqind-1]:sp[eqind+1]})
        if g.startswith(' Current'): current.extend([float(sp) for sp in g.split()[3:]])
        elif g.startswith(' Bunch'): blen.extend([float(sp) for sp in g.split()[4:]])
        elif g.startswith(' Synchrotron'): nus.extend([float(sp) for sp in g.split()[3:]])
        elif g.startswith(' (Nu-Nux)'):
            g = f.next()
            ind = 0
            while not g.isspace():
                test = g.replace(':',' ').split()
                start = 0
                if firstthru: ln = []
                for i,t in enumerate(test):
                    if firstthru: print t
                    if i%2: start += 1j*float(t)
                    else: start += float(t)
                    if type(start)==complex: 
                        if firstthru: ln.append(start)
                        else: line[ind].append(start)
                        start = 0
                if firstthru: line.append(ln)
                ind += 1
                g = f.next()
            firstthru = False
        elif g.startswith('1PROBLEM'): break
    return valuedict, np.array(line), current, blen, nus

def calcZEff(impmodel,sring,plane='V',truncate=10000):
    """
    Calculate the effective impedance of a storate given an impedance model.
    *impmodel* - eigen_system.ImpedanceModel class
    *sring* - utility.StorageRing class
    *plane* - Which plane to use, relevant for nonero chromaticity
    *truncate* - Truncate the integration at this number of revolution harmonics
    """
    revfreq = sring.frf/sring.nbunch
    freq = np.r_[-truncate:0,1:truncate]*revfreq
    #freq = np.arange(-truncate,truncate+0.5,1)*revfreq
    if plane=='V': chrofreq = revfreq*sring.vchro/sring.alphac
    elif plane=='H': chrofreq = revfreq*sring.hchro/sring.alphac
    bspect = np.exp(-(2*np.pi*(freq-chrofreq)*sring.blen)**2)

    return np.trapz(bspect*impmodel.calcImpedance(freq))/np.trapz(bspect)

def dispersionIntegral(ksi,b):
    """
    Calculate the Chin dispersion integral according to the formulation in the Chin SPS note: CERN SPS /5-9
    *ksi* - The coherent tune shift normatlised by the tune spread parameter (can be an array)
    *b* - The tune shift at the square of the RMS beam size (factor of two compared to beam size at <J>)
    """
    const = 1/2./b
    if np.all(np.imag(ksi)==0): ksi = ksi+np.sign(b)*1e-15*1j
    result = -const*(1-ksi*np.exp(-ksi,dtype=np.complex128)*(special.expi(ksi,dtype=np.complex128)-np.sign(b)*1j*np.pi))        
    return result

def calcQso(sr):
    return np.sqrt(sr.nbunch*sr.alphac*sr.vrf*np.sqrt(1-(sr.eloss/sr.vrf)**2)/2/np.pi/sr.energy)
