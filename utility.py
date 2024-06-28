import subprocess as sb
import numpy as np
from numpy import ma
#from IPython.core.debugger import Tracer

def getOutput(cmdstr,stdin=None,stderr=None):
    """
    get output from single command string or list
    """

    if not isinstance(cmdstr,list): cmdstr = cmdstr.split()
    if '|' in cmdstr:
        splitind = cmdstr.index('|')
        p = sb.Popen(cmdstr[:splitind],stdout=sb.PIPE)
        return getOutput(cmdstr[splitind+1:],p.stdout)

    return sb.Popen(cmdstr,stdout=sb.PIPE,stdin=stdin).communicate()[0]

def convergenceTest(seq,nullres=5):
    """
    determine when a sequence converges to an exact number of significant figures
    """
    pass

def sort2D(arr,order=[]):
    """
    Use numpy.sort to sort any array with numbers in the order instead of fields
    """
    
    strarr = np.array(zip(*list(arr)),[(str(n),float) for n in range(arr.shape[0])])
    output = np.sort(strarr,order=[str(o) for o in order])
    output = np.array([output[o[0]] for o in output.dtype.names])
    return output

def saveData(filename,array,header,**kwargs):
    """
    Use numpy.savetxt to save an *array* to *filename* with an additional *header*
    """
    if not header.endswith('\n'): header += '\n'
    f = open(filename,'w')
    f.write(header)
    f.close()
    f = open(filename,'a')
    np.savetxt(f,array,**kwargs)
    f.close()

def load3D(filename,turns=None,return_comments=False,**kwargs):
    """
    Load a 2 dimensional array from *filename* that stores data from a number of *turns*.
    The third dimension is split by commented lines whose first character can be specified
    by passing the keyword arguments for numpy.loadtxt
    """

    if 'skiprows' not in kwargs: kwargs.update({'skiprows':2})
    if 'comments' not in kwargs: kwargs.update({'comments':'\n'})

    data = np.loadtxt(filename,**kwargs)

    if turns==None or return_comments:
        file3d = open(filename,'r')
        turns = 0
        comms = []
        for f in file3d:
            if f.startswith(kwargs['comments']):
                comms.append(f)
                turns+=1
        file3d.close()

    print(filename, turns)
    try: data = data.reshape(turns,-1,data.shape[-1])
    except(ValueError): data = data.reshape(turns+1,-1,data.shape[-1])

    if return_comments: return data.T, comms
    return data.T

def scientificMultipleLabel(ax,order,axis='x'):

    from matplotlib import rcParams
    
    label=r'$\times10^{%d}$' % order
    if axis=='x':
        bot = ax.figure.subplotpars.bottom    
        rig = ax.figure.subplotpars.right    
        ax.annotate(label,(rig,0),ha='right',va='bottom',
                    size=rcParams['xtick.labelsize'],xycoords='figure fraction')
    if axis=='y':
        top = ax.figure.subplotpars.top
        lef = ax.figure.subplotpars.left
        ax.annotate(label,(lef,top),ha='left',va='bottom',
                    size=rcParams['ytick.labelsize'],xycoords='figure fraction')
    if axis=='y2':
        top = ax.figure.subplotpars.top
        rig = ax.figure.subplotpars.right
        ax.annotate(label,(rig,top),ha='right',va='bottom',
                    size=rcParams['ytick.labelsize'],xycoords='figure fraction')

def propAndFid(x,y,xerr,yerr):

    pfit = np.polyfit(x,y,1)
    terr = np.sqrt(yerr**2+(pfit[0]*xerr)**2)
    
    return chiFit(x,y,terr)

def chiFit(x,y,yerr):
    """
    Returns (gradient, intercept), (gradient error, intercept error), chi squared per degree of freedom
    """

    weights=1/yerr**2
    xbar = np.average(x,weights=weights)
    ybar = np.average(y,weights=weights)

    normconst = np.sum(weights*(x-xbar)**2)
    m = np.sum(weights*(x-xbar)*(y-ybar))/normconst
    c = ybar-m*xbar

    return np.array([m,c]), np.array([np.sqrt(1/normconst), np.sqrt(np.sum(weights)+xbar**2/normconst)]), np.sum(weights*(y-m*x-c)**2)/(len(y)-2)

def scatterFit(x,y):

    n = float(len(y))
    pf = np.polyfit(x,y,1)
    ss = np.sqrt(np.sum((y-np.polyval(pf,x))**2)/(n-2.))
    sumx2 = np.sum(x**2)
    denom = 1/np.sqrt(n*sumx2-np.sum(x)**2)
    erra = ss*np.sqrt(n)*denom
    errb = ss*np.sqrt(sumx2/(n*sumx2-np.sum(x)**2))

    return pf, erra, errb

def zeroSlopeFit(x,y,full_output=False):

    lenny = len(y)
    res = np.zeros(lenny-2)
    fullout = np.zeros((lenny-2,lenny))
    for n in range(2,lenny,1):
        pfit = np.polyfit(x,y,n-1)
        res[n-2] = pfit[-2]
        fullout[n-2,-n:] = pfit

    if full_output: return res, fullout
    else: return res

def loadArchiveTool(filename):
    
    import time
    
    datefmt = '%Y-%m-%d_%H:%M:%S.%f'
    convtime = lambda x: time.mktime(time.strptime(x,datefmt))

    return np.loadtxt(filename,converters={0:convtime,1:float},comments='"')

class StorageRing(object):

    def __init__(self,filename=''):
        """
        Class to hold almost every parameter of a storage ring possible, defaulted to None
        """

        self.name = None
        self.energy = None
        self.nbunch = None
        self.blen = None
        self.revtime = None
        self.eloss = None
        self.vtune = None
        self.htune = None
        self.ltune = None
        self.vchro = None
        self.hchro = None
        self.hfrac = None
        self.hphi = None
        self.phi0 = None
        self.vrf = None
        self.betay = None
        self.frf = None

        if filename:
            self.filename = filename
            if self.filename.endswith('.inp'): self._readMachineFile()        

    def _readMachineFile(self):

        f = open(self.filename,'r')
        indict = {}
        for e in f:
            if e.startswith('#'): continue
            e = e.split('=')
            try: indict.update({e[0].strip():eval(e[1])})
            except(NameError): indict.update({e[0].strip():e[1].strip()})
        self.__dict__.update(indict)
        f.close()

class LoadTxt:

    def __init__(self,filename,*args,**kwargs):

        self.data = np.loadtxt(filename,*args,**kwargs)
        gdat = open(filename,'r')
        self.gdat = []
        for g in gdat:
            if not g.startswith('#'): break
            if '=' in g: self.gdat.append(eval("{'"+g[1:].replace('=',"':").replace(',',",'")+'}'))
            else: self.gdat.append(eval('['+g[1:]+']'))

class ResultsDir:

    def __init__(self,dirname,clsname,*args,**kwargs):
        """
        Load several instances of a class with name *clsname* from subdirectories
        in directory *dirname*. *args and **kwargs correspond to the arguments and
        keyword arguments of the class constructor respectively.
        """

        if 'pattern' not in kwargs:
            self.pattern = ''
        else:
            self.pattern = kwargs.pop('pattern')
        if 'init' not in kwargs:
            self.init = True
        else:
            self.init = kwargs.pop('init')
        if 'filetype' not in kwargs:
            self.ftype = '[@/]'
        else:
            self.ftype = kwargs.pop('filetype')

        #self.dirs = filter(lambda x: self.pattern in x, getOutput('ls -F '+dirname+' | grep [/@]').replace('@','/').split('\n')[:-1])
        self.dirs = getOutput('ls -F '+dirname+' | grep '+self.pattern+self.ftype)
        self.dirs = self.dirs.replace('@','/').replace('*','').split('\n')[:-1]
        self.results = []
        self.dirname = dirname+int(not dirname.endswith('/'))*'/'
        
        if self.init:
            self.initialise(clsname,*args,**kwargs)

    def initialise(self,clsname,*args,**kwargs):
        for d in self.dirs:
            print(d)
            self.results.append(clsname(self.dirname+d,*args,**kwargs))

    def callMethod(self,method,*args,**kwargs):
        """
        Call class method *method* for all the stored instances with the same *args
        and **kwargs.
        """
        results = []
        for r,d in zip(self.results,self.dirs):
            print('ResultsDir> calling for ', d)
            results.append(method(r,*args,**kwargs))
        return results

    def sort(self,vals):
        """
        Sort results in the same order as increasing *vals*, originally in corresponding order to dirs
        """

        tmp_arr = np.array(zip(np.arange(len(vals)),vals),dtype=[('inds',int),('vals',float)])
        tmp_arr.sort(order=['vals'])
        tmp_res = []
        tmp_dirs = []
        for r in tmp_arr['inds']:
            tmp_res.append(self.results[r])
            tmp_dirs.append(self.dirs[r])

        self.results = tmp_res
        self.dirs = tmp_dirs

def integrate(data,cumulative=False):

    if cumulative:
        output = np.zeros(len(data.T))
        output[1:] = np.cumsum((data[1,:-1]+data[1,1:])/2.*(data[0,1:]-data[0,:-1]))
        return output
    else:
        return np.trapz(data[1],data[0])

class MatlabData(object):

    def __init__(self,file):

        from scipy import io

        self._file = file
        self._data = io.loadmat(self._file)

        self.construct(self._data)

    def construct(self,data,prefix=''):

        for d in data:
#            if len(data[d[0]].dtype)!=0:
            if isinstance(data[d],dict):
                self.construct(data[d[0]],prefix=d[0]+'.')
            else:
                if isinstance(data[d],np.ndarray):
                    self.__setattr__(prefix+d,data[d].squeeze())
                else:
                    self.__setattr__(prefix+d,data[d])

    def splitAll(self,refname,*args,**kwargs):

        ref = self.__getattribute__(refname)
        lenref = len(ref)
        for k,d in self.__dict__.iteritems():
            if len(d)==len(ref):
                self.splitAxis(k,*args,**kwargs)

    def splitAxis(self,attrname,newlength,axis=0,order='C'):

        if isinstance(attrname,(list,tuple,np.ndarray)):
            for a in attrname:
                self.splitAxis(a,newlength,axis,order)
            return

        tobereshaped = self.__getattribute__(attrname)
        newshape = list(tobereshaped.shape)
        if axis==0:
            tobereshaped = tobereshaped[:(newshape[0]/newlength)*newlength]
        newshape.insert(axis,newshape[axis]/newlength)
        newshape[axis+1] = newlength
        
        self.__setattr__(attrname,tobereshaped.reshape(newshape))        

    def meanAndError(self,attrname,axis=-1,qcut=0,filt=None):

        attr = self.__getattribute__(attrname)
        if qcut>0:
            trlist = range(attr.ndim)
            trlist[trlist.index(axis)] = 0
            trlist[0] = axis
            tmp_arr = attr.transpose(*trlist)
            attr = ma.array(attr,mask=~quartileCut(tmp_arr).transpose(trlist))
        if np.all(filt==None):
            self.__setattr__(attrname+'bar',ma.mean(attr,axis=axis))
            self.__setattr__(attrname+'err',ma.std(attr,axis=axis,ddof=1)/np.sqrt(attr.shape[axis]))
        else:
            if hasattr(attr,'mask'):  attr.mask += ~filt
            else: attr = ma.array(attr,mask=~filt)
            meanmask = np.prod(attr.mask,axis=axis)
            self.__setattr__(attrname+'bar',ma.array(ma.mean(attr,axis=axis),mask=meanmask))
            self.__setattr__(attrname+'err',ma.array(ma.std(attr,axis=axis,ddof=1)/np.sqrt(attr.shape[axis]),mask=meanmask))

    def groupBy(self,valname,attrname,vals,tol=1e-3,consecutive=False):

        valign = self.__getattribute__(valname)
        attr = self.__getattribute__(attrname)
        valres = []
        attrres = []
        maxlen = 0

        ######This part is a complicated fudge for same currents with different currents per bunch
        ccount = {}
        if consecutive:
            for i,v in enumerate(vals):
                cfilt = np.absolute(v-vals)<tol
                if np.sum(cfilt)>1:
                    ccount.update({i:np.where(vals[cfilt]==v)[0][0]})
        
        for i,v in enumerate(vals):
            filt = np.absolute(valign-v)<tol
            if np.sum(filt)>maxlen:
                maxlen = np.sum(filt)
                
            if i in ccount:
                inds = np.where(filt)[0]
                ind_breaks = np.where((inds[1:]-inds[:-1]>1) | (inds[1:]==inds[-1]))[0]+1
                if ccount[i]>0:
                    filt[:inds[ind_breaks[ccount[i]-1]]] = False
                filt[inds[ind_breaks[ccount[i]]]:] = False
                #Tracer()()
            
            attrres.append(attr[filt])
            valres.append(valign[filt])
            
        res = ma.array(np.zeros((len(vals),maxlen)),mask=np.zeros((len(vals),maxlen),bool))
        val = ma.array(np.zeros((len(vals),maxlen)),mask=np.zeros((len(vals),maxlen),bool))
        for i,(a,v) in enumerate(zip(attrres,valres)):
            res[i][:len(a)] = a
            res.mask[i][len(a):] = True
            val[i][:len(v)] = v
            val.mask[i][len(v):] = True            
        self.__setattr__(attrname+'_by'+valname,res)
        self.__setattr__(valname+'_with'+attrname,val)

def quartileCut(arr,nquarts=5):

    med = np.percentile(arr,50,0)
    upper_quartile = np.percentile(arr,75,0)
    lower_quartile = np.percentile(arr,25,0)
    
    return (arr-med<=nquarts*(upper_quartile-med)) & (med-arr<=nquarts*(med-lower_quartile))
        
def findPeakCentre(dat,fraction=0.5,axis=-1):

    data = dat-np.mean(dat[-10:],axis=axis)
    pkmax = np.amax(data,axis=axis)
    pkpos = np.argmax(data,axis=axis)
    fracbefore = np.argmin(np.absolute(data[:pkpos]-pkmax),axis=axis)
    fracafter = np.argmin(np.absolute(data[pkpos:]-pkmax),axis=axis)+pkpos

    return (fracbefore+fracafter)/2.

def loadTaurusFile(fname,archiver=False):

    import time

    def conv(x):
        #if archiver:
        #    
        #else:
        #    x = x.split('.')
        #    ret = time.mktime(time.strptime(x[0],'%Y-%m-%d_%H:%M:%S'))+float('.'+x[1])
        uncnv = ''
        while True:
            try:
                ret = time.mktime(time.strptime(x,'%Y-%m-%d_%H:%M:%S'))
            except(ValueError) as v:
                if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
                    uncnv += x[-1]
                    x = x[:-1]
                else:
                    raise
            else:
                break
        uncnv = uncnv[-1::-1]
        if not uncnv.startswith('.'): ret += float('.'+uncnv)
        else: ret += float('.'+uncnv)
            
        return ret
    
    return np.loadtxt(fname,converters={0:conv,1:float})

def simpleRectify(wfar):
    """
    Determine the signal envelope my sampling the maximum signal between zero crossings
    """

    import itertools

    sig = np.sign(wfar)
    zerocross = zip(*np.nonzero(np.array(sig[:,:-1]-sig[:,1:],bool) & (sig[:,:-1]!=0)))

    roof = []
    poss = []
    for k,z in itertools.groupby(zerocross, lambda x: x[0]):
        rf = []
        ps = []
        zd = z.next()[1]
        for zu in z:
            sect = np.absolute(wfar[zu[0],zd:zu[1]])
            loc = np.argmax(sect)
            rf.append(sect[loc])
            ps.append(zd+loc)
            zd = 1*zu[1]
        roof.append(rf)
        poss.append(ps)
            
    return poss,roof

def naffSimple(arr,*args,**kwargs):

    if 'axis' in kwargs and kwargs['axis']!=0:
        axisToZero(arr,kwargs['axis'])
    if 'dcReject' in kwargs and kwargs['dcReject']:
        arr = arr-np.mean(arr,0)
    if 'minfreq' in kwargs:
        min_ind = kwargs['minfreq']*len(arr)
    else: min_ind = 0
    if 'freqsteps' in kwargs:
        freqsteps = kwargs['freqsteps']
    else:
        freqsteps = 1024
    if 'complete' in kwargs:
        complete = kwargs['complete']
    else:
        complete = False
        
    lenarrf = float(len(arr))
    ini_fft = np.absolute(np.fft.fft(arr,axis=0))
    ini_guess = (np.argmax(ini_fft[min_ind:len(arr)/2,:],axis=0)+min_ind)/lenarrf
    frrnge = np.linspace(-0.5,0.5,freqsteps)/lenarrf
    freqrange = np.ones((len(ini_guess),freqsteps))*frrnge
    freqrange = (freqrange.T+ini_guess).T
    
    rnge = np.ones(np.shape(arr)[-1::-1])*np.arange(len(arr))
    tmplt = np.ones((freqsteps,)+np.shape(rnge))*rnge
    tmplt = (tmplt.T*freqrange).T#shape (1024, 201, 2048)
    freqscan = np.exp(1j*2*np.pi*tmplt)
    pkloc = np.argmax(np.absolute(np.sum(freqscan*arr.T,axis=2)),axis=0)
    freqshift = frrnge[pkloc]
    complex_peak = np.sum(freqscan*arr.T,axis=2)[pkloc,np.arange(len(pkloc))]

    if complete:
        freqrange = np.linspace(0,1,arr.shape[0])+freqshift
        tmplt = np.ones((arr.shape[0],)+np.shape(rnge))*rnge
        tmplt = (tmplt.T*freqrange).T
        freqscan = np.exp(1j*2*np.pi*tmplt)

        return ini_guess+freqshift, complex_peak, np.sum(freqscan*arr.T,axis=2)
    
    return ini_guess+freqshift, complex_peak

def acceptReject(nsamples,rmin,rmax,distfunction):
    """
    Generate a distribution of *nsamples* between rmin and rmax
    according to arbitrary callback *distfunction*, whose range
    is between 0 and 1 (algorithm is most efficient if maximum
    is exactly 1).
    """
    good_samples = 0
    pot = np.random.rand(2,2*nsamples)
    pot[0] = (pot[0]-0.5)*(rmax-rmin)+(rmax+rmin)/2.0
    distout = np.zeros(nsamples)
    ind = 0
    while good_samples<nsamples:
        if pot[1,ind]<distfunction(pot[0,ind]):
            distout[good_samples] = pot[0,ind]
            good_samples+=1
        ind+=1
        if ind>2*nsamples-1:
            pot = np.random.rand(2,2*nsamples)
            pot[0] = pot[0]*(rmax-rmin)-rmax+(rmax+rmin)/2.0
            ind = 0
    return distout

def fitHyperDecay(xd,yd):

    from scipy import optimize

    hyperdecay = lambda y0,tau: y0/(1+xd/tau)
    chi2 = lambda x: np.sum((yd-hyperdecay(x[0],x[1]))**2)

    opt = optimize.fmin(chi2,[np.amax(yd),np.mean(xd)])

    return opt

def unrollPhase(ph):
    resarr = ph.copy()
    urptsm = np.where(ph[1:]-ph[:-1]<-np.pi)[0]
    urptsp = np.where(ph[1:]-ph[:-1]>np.pi)[0]
    for u in urptsm:
        resarr[u+1:] += 2*np.pi
    for u in urptsp:
        resarr[u+1:] -= 2*np.pi

    return resarr

def calcQso(sring):

    return np.sqrt(sring.nbunch*sring.alphac*sring.vrf*np.sqrt(1-sring.eloss**2/sring.vrf**2)/2./np.pi/sring.energy)

def valueGrid(x,y,z):
    """
    X is a 1D array for one of the axes
    Y is a 2D array of the same length as x and increasing along its other dimension
    Z is a 2D array of the same shape as Y
    """

    mindiffx = np.nanmin(np.absolute(x[1:]-x[:-1]))
    mindiffy = np.nanmin(np.absolute(y[:,1:]-y[:,:-1]))
    gridx = np.arange(np.nanmin(x),np.nanmax(x)+mindiffx/2.,mindiffx)
    gridy = np.arange(np.nanmin(y),np.nanmax(y)+mindiffy/2.,mindiffy)

    arr = np.ones((len(gridx),len(gridy)))*np.nan
    last_cx = 0
    last_cy = 0
    for xa,ya,za in zip(x,y,z):
        cx = np.nanargmin(np.absolute(gridx-xa))
        for yb,zb,yc in zip(ya[:-1],za[:-1],ya[1:]):
            if not np.isnan(yb):
                cy = np.nanargmin(np.absolute(gridy-yb))
            else:
                continue
            if not np.isnan(yc):
                next_cy = np.nanargmin(np.absolute(gridy-yc))
            patchlen = min(cy-last_cy,next_cy-cy+np.nan*(next_cy==cy))
            arr[cx,cy-patchlen/2:] = zb
            last_cy = 1*cy
        end_ind = min(next_cy+(next_cy-cy)/2,arr.shape[1])
        print(next_cy, cy)
        print(next_cy+(next_cy-cy)/2, arr.shape[1], end_ind)
        arr[cx,(next_cy+cy)/2:end_ind] = za[-1]
        arr[cx,end_ind+1:] = np.nan
        last_cy = 0

    return gridx, gridy, arr

            
            
    

