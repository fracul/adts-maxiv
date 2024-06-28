import numpy as np
import utility
from pylab import figure, setp, text
import eigen_system
import matplotlib as mpl
import matplotlib.colors as mcolors

s = utility.StorageRing('maxiv.inp')
adts1000 = 1000/s.betay

def prepCompPlot(twoMode=False):
    """
    Do the necessary calculations to produce plots of the eigenvalues for different beam currents to be compared with the Landau contours.
    *twoMode* - Limit the calculation to the zeroth head-tail mode and the -1 head-tail mode.
    Returns the eigen_system.EigenSystem and eigen_system.Scan instances of a current scan.
    """

    azimodes = 2-twoMode
    radmodes = 1
    
    i = eigen_system.ImpedanceModel(rfreq=11e9,rs=200e3)
    e = eigen_system.EigenSystem(s,i,30000,azimodes,radmodes,radimin=0,plane='Y',precision128=True)
    if twoMode:
        e.azimax = 1
        e.azicount = e.azimax-e.azimin
        e.nmodes = radmodes*e.azicount
    currs = np.arange(0.00001,0.01,0.0001)
    sc = eigen_system.Scan(e,{'current':np.array(currs)})
    sc.scan()

    return e, sc

def prepDelphiBMark():
    """
    Do the necessary calculations to produce plots of the eigenvalues and dispersion integrals for comparison with the CERN DELPHI code and similiarly.
    Returns the eigen_sytesm.EigenSystem class and an eigen_system.Scan classes for the current scan.
    """

    azimodes = 2
    radmodes = 3
    
    i = eigen_system.ImpedanceModel(rfreq=11e9,rs=200e3)
    e = eigen_system.EigenSystem(s,i,30000,azimodes,radmodes,radimin=0,plane='Y',precision128=True)
    #e.azimax = 1
    #e.azicount = e.azimax-e.azimin
    #e.nmodes = radmodes*e.azicount
    currs = np.arange(0.00001,0.01,0.0001)
    sc = eigen_system.Scan(e,{'current':np.array(currs)})
    sc.scan()

    return e, sc

def landauContours(scanres,signb=1,csinv=False):
    """
    Plot the Landau contours along with the results of the eigenvalue calculationss.
    """

    f = figure()
    f.subplots_adjust(left=0.2)
    ax = f.add_subplot(111)

    ksi = np.arange(-50,50,0.01)
    convfact = s.ltune*scanres.esys.revfreq
    imconst = (2*np.pi*scanres.esys.revfreq/1e3)**csinv#Convert to tune values if plotting in terms of action J
    reconst = (scanres.esys.revfreq)**csinv#Convert to tune values if plotting in terms of action J    
    cmap = ax.scatter(scanres.results.real*convfact/reconst,scanres.results.imag*2*np.pi*convfact/imconst,cmap=mpl.cm.plasma,
                      marker='.',c=(np.ones(scanres.results.shape).T*scanres.scan_vals['current']*1e3).T,zorder=1,
                      vmin=0,vmax=10)
    if csinv:
        ax.set_xlabel('Coherent tune shift')
        ax.set_ylabel(r'Growth rate (${\rm turn}^{-1}$)')
    else:
        ax.set_xlabel('Coherent frequency shift (Hz)')
        ax.set_ylabel(r'Growth rate ($\rm s^{-1}$)')

    if csinv:
        bs2 = np.arange(0.05,1.025,0.05)*5/2.*imconst
    else:
        bs2 = np.arange(500,10250,500)        
    if signb<0:
        bsigma2 = bs2/5
    else:
        bsigma2 = np.zeros(len(bs2)+1)
        bsigma2[0] = 100
        bsigma2[1:] = bs2
    landau_contours = 1/np.array([eigen_system.dispersionIntegral(ksi,signb*b) for b in bsigma2])
    t_bwr = mpl.cm.plasma
    
    for i,(l,b) in enumerate(zip(landau_contours,bsigma2)):
        cnum = i/float(landau_contours.shape[0])
        ax.plot(l.real/2/np.pi/reconst,l.imag/imconst,'--k',zorder=-1)#c=t_bwr(cnum),ls='--')
        if csinv:
            txtstr = '%.2f' % (signb*b*2/imconst)
        else:
            txtstr = '%d' % (signb*b)
        if signb<0:
            if i>1 and (i-1)%2 and i<7:
                maxarg = np.argmax(l.imag)
                txt = text((l.real[maxarg]+300)/2/np.pi/reconst,(l.imag[maxarg]-50)/imconst,txtstr,ha='right',va='center')
                txt.set_bbox(dict(facecolor='w',alpha=1.0,edgecolor='w'))                
        else:
            if i>1 and (i-1)%2 and i<15:
                y = 1200
                filt = l.real<0
                xarg = np.argmin(np.absolute(l.imag[filt]-y))
                if i>3 and l.imag[filt][xarg]>y:
                    x = l.real[filt][xarg:xarg+2]/2/np.pi
                    xfine = np.arange(x[0],x[1],(x[1]-x[0])/100.)
                    yinterp = np.interp(xfine,x[-1::-1],l.imag[filt][xarg:xarg+2][-1::-1])
                    xreal = xfine[np.argmin(np.absolute(yinterp-y))]                    
                else:
                    x = l.real[filt][xarg-1:xarg+1]/2/np.pi
                    xfine = np.arange(x[0],x[1],(x[1]-x[0])/100.)
                    yinterp = np.interp(xfine,x[-1::-1],l.imag[filt][xarg-1:xarg+1][-1::-1])
                    xreal = xfine[np.argmin(np.absolute(yinterp-y))]
                print i, x, l.imag[filt][xarg-1:xarg+1], xreal
                #print xfine
                txt = text(xreal/reconst,y/imconst,txtstr,ha='center',va='center',rotation=78)
                txt.set_bbox(dict(facecolor='w',alpha=1.0,edgecolor='w'))                

    cbar = f.colorbar(cmap)
    cbar.set_label('Bunch current (mA)',rotation=270,va='bottom')
    
    ax.set_xlim(-2200/reconst,400/reconst)
    if not csinv:
        ax.set_ylim(-1000,1900)
        ax.set_xticks(np.arange(-2000,10,1000))
    else:
        ax.set_ylim(-0.3,0.4)
        ax.set_yticks(np.arange(-0.2,0.45,0.2))
        ax.set_xticks(np.arange(-0.004,0.0011,0.002))
        utility.scientificMultipleLabel(ax,-3,axis='y')        

    return landau_contours
    
def delphiCompareSuite(*delphiBMOut):
    """
    Produce plots of a comparison with the CERN DELPHI code.
    """

    f = figure()
    ax = f.add_subplot(111)

    ksi = np.arange(-50,50,0.01)
    lc = eigen_system.dispersionIntegral(ksi,500.)
    lcm = eigen_system.dispersionIntegral(ksi,-500.)
    tst = np.genfromtxt('dispintegralInverse_adts1000_DELPHI_allpositive_positiveADTS.txt',dtype=complex)
    tstm = np.genfromtxt('dispintegralInverse_adts1000_DELPHI_allpositive_negativeADTS.txt',dtype=complex)     

    esys = delphiBMOut[0]
    scan = delphiBMOut[1]

    ax.plot((1/lc).real/2/np.pi,(1/lc).imag,'-C0',ms=4)    
    ax.plot(tst.real[::4]*esys.revfreq,tst.imag[::4]*2*np.pi*esys.revfreq,'.C0')
    ax.plot((1/lcm).real/2/np.pi,(1/lcm).imag,'-C1',ms=4)
    ax.plot(tstm.real[::4]*esys.revfreq,tstm.imag[::4]*2*np.pi*esys.revfreq,'.C1')    
    ax.set_ylim(0,1100)
    ax.set_xlim(-3000,3000)

    ax.set_xlabel('Coherent frequency shift (Hz)')
    ax.set_ylabel(r'Growth rate ($\rm s^{-1}$)')

    art = ax.legend(('In house','DELPHI'))
    ax.legend(ax.lines[::2],('Positive\nADTS','Negative\nADTS'),loc=2)
    ax.add_artist(art)
    ax.set_xticks(np.arange(-3000,3010,1500))

    f = figure()
    f.subplots_adjust(left=0.21,right=0.95)    
    ax = f.add_subplot(111)

    delphires = np.genfromtxt('eigenvalues_fromBeta_DELPHI.txt',dtype=complex)
    ax.plot(scan.scan_vals['current']*1e3,scan.results.real*s.ltune*esys.revfreq,'.C0',label='In house')
    ax.plot(1e3*delphires[:,0].real,delphires[:,1:].real/2/np.pi,'.C1',ms=8,label='DELPHI')

    ax.set_xlabel('Bunch current (mA)')
    ax.set_ylabel('Coherent frequency offset (Hz)')

    #ax.legend()
    skip = scan.results.shape[1]    
    ax.legend(ax.lines[::skip],('In house','DELPHI'),loc=2,ncol=2)
    ax.set_ylim(-2500,2500)
    ax.set_xlim(0,10)

    f = figure()
    f.subplots_adjust(left=0.21,right=0.95)
    ax = f.add_subplot(111)

    ax.plot(scan.scan_vals['current']*1e3,scan.results.imag*s.ltune*esys.revfreq*2*np.pi,'.C0',label='In house')
    ax.plot(1e3*delphires[:,0].real,delphires[:,1:].imag,'.C1',ms=8,label='DELPHI')

    ax.set_xlabel('Bunch current (mA)')
    ax.set_ylabel('Coherent frequency offset (Hz)')

    #ax.legend()
    ax.legend(ax.lines[::skip],('In house','DELPHI'))
    ax.set_xlim(0,10)
    

    
            

        
    
    
    

    
    
