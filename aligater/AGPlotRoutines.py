#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#			~~aliGater~~
#	(semi)automated gating software
#
#               /^^\
#   /^^\_______/0  \_
#  (                 `~+++,,_________,,++~^^^^^^^
#..V^V^V^V^V^V^\.................................
#
#
#	Parsing flow data with fcsparser from Eugene Yurtsevs FlowCytometryTools (very slightly modified)
#	Check out his excellent toolkit for flow cytometry analysis: 
#	http://eyurtsev.github.io/FlowCytometryTools/
#
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	https://www.med.lu.se/labmed/hematologi_och_transfusionsmedicin/forskning/bjoern_nilsson

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Ellipse, Arrow
from matplotlib.ticker import Locator, Formatter
import six
from matplotlib import rcParams

#AliGater imports
import aligater.AGConfig as agconf
from aligater.AGFileSystem import getGatedVector, AliGaterError

from scipy.ndimage.filters import gaussian_filter1d
import sys

sentinel = object()


    
    
def plotHeatmap(fcsDF, x, y, vI=sentinel, bins=300, scale='linear', xscale='linear', yscale='linear', thresh=1000, aspect='auto', **kwargs):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)<2:
        sys.stderr.write("Passed index contains no events\n")
        return None, None
    if len(vI)<bins:
        bins=len(vI)
    if scale.lower()=='logish':
        xscale='logish'
        yscale='logish'
    #Default x and y lims        
    bYlim=False
    bXlim=False
    #TODO:testing for auto limits
    if 'xlim' in kwargs:
            if not isinstance(kwargs['xlim'],list):
                raise TypeError("if xlim is passed, it must be a list of float/int")
            elif not all(isinstance(i,(float,int)) for i in kwargs['xlim']):
                raise TypeError("Non float/int element encountered in xlim")
            else:
                xscale_limits=kwargs['xlim']
                if xscale.lower()=='logish':
                    xscale_limits=logishTransform(xscale_limits,thresh)
                    bXlim=True
    if 'ylim' in kwargs:
            if not isinstance(kwargs['ylim'],list):
                raise TypeError("if ylim is passed, it must be a list of float/int")
            elif not all(isinstance(i,(float,int)) for i in kwargs['ylim']):
                raise TypeError("Non float/int element encountered in ylim")
            else:
                yscale_limits=kwargs['ylim'] 
                if yscale.lower()=='logish':
                    yscale_limits=logishTransform(yscale_limits,thresh)
                    bYlim=True
                    
    vX=getGatedVector(fcsDF, x, vI)
    vY=getGatedVector(fcsDF, y, vI)
    plt.clf()
    matplotlib.rcParams['image.cmap'] = 'jet'
    heatmap, xedges, yedges = getHeatmap(vX, vY, bins, scale, xscale, yscale, thresh)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap=np.ma.masked_where(heatmap == 0, heatmap)
    plt.clf()
    fig, ax = plt.subplots()
    plt.imshow(heatmap.T, extent=extent, origin='lower',aspect=aspect)
    plt.xlabel(x)
    plt.ylabel(y)
    cmap=plt.get_cmap()
    cmap.set_bad(color='white') #Zeroes should be white, not blue

    if xscale.lower()=='logish':
        ax=plt.gca()
        ax.xaxis.set_major_locator(LogishLocator(subs='all'))
        ax.xaxis.set_major_formatter(LogishFormatter())
    if yscale.lower()=='logish':
        ax=plt.gca()
        ax.yaxis.set_major_locator(LogishLocator(subs='all'))
        ax.yaxis.set_major_formatter(LogishFormatter())
    if bXlim:
        ax.xaxis.set_xlim(left=xscale_limits[0], right=xscale_limits[1])
    if bYlim:
        ax.yaxis.set_xlim(left=yscale_limits[0], right=yscale_limits[1])
    return fig,ax

def getHeatmap(vX, vY, bins, scale, xscale, yscale, T=1000, normalize=False, xlim=None, ylim=None, range=None):
    if not any(isinstance(i,str) for i in [scale,xscale,yscale]):
        raise TypeError("scale, xscale, yscale must be specified as string, such as: 'linear', 'logish'")
    if not all(i in ['linear', 'logish'] for i in [scale,xscale,yscale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logish'")
    if range is not None:
        if isinstance(range,list):
            if len(range)==2:
                if not all(isinstance(i,(list)) for i in range):
                    AliGaterError("in getHeatmap, invalid dtype encountered in range, expected two list-likes")
                else:
                    if not all(isinstance(i,(float,int)) for i in range[0]) or not all(isinstance(i,(float,int)) for i in range[1]):
                        AliGaterError("in getHeatmap,invalid dtype encountered in range")
                    else:
                        defaultRange=range
                        xRange=range[0]
                        yRange=range[1]
            else:
                AliGaterError("in getHeatmap, range must be list, found "+str(type(range)))
        else:
            AliGaterError("in getHeatmap, custom range passed but is not list, found type: "+str(type(range)))
    else:
        defaultRange=None
        xRange=None
        yRange=None
    assert len(vX) == len(vY)
    index_mask=[]
    for i in np.arange(len(vX)-1,-1,-1):
        if i < 0:
            #TODO
            raise
        if xlim is not None:
            if vX[i] < xlim[0] or vX[i] > xlim[1]:
                index_mask.append(i)
                continue
            if vY[i] < ylim[0] or vY[i] > ylim[1]:
                index_mask.append(i)
    if len(index_mask) > 0:
        vX = np.delete(vX, index_mask)
        vY = np.delete(vY, index_mask)
        assert len(vX) == len(vY)
                
    if scale=='linear' and xscale=='linear' and yscale == 'linear':
        return np.histogram2d(vX, vY, bins, normed=normalize, range=defaultRange)
    elif scale=='logish' or (xscale == 'logish' and yscale == 'logish'):
        xBinEdges=logishBin(vX,bins,T, xRange)
        yBinEdges=logishBin(vY,bins,T, yRange)
        return np.histogram2d(vX, vY, [xBinEdges,yBinEdges])
    if xscale=='logish':
        xBinEdges=logishBin(vX,bins,T, xRange)
        return np.histogram2d(vX, vY, [xBinEdges,bins])
    if yscale=='logish':
        yBinEdges=logishBin(vY,bins,T, yRange)
        return np.histogram2d(vX, vY, [bins,yBinEdges])

def logishBin(vX, bins, T, customRange=None):
    if customRange is not None:
        defaultRange=customRange
    else:
        defaultRange=[min(vX),max(vX)]
    transformedRange=logishTransform(defaultRange,T)
    transformedBinEdges=np.linspace(transformedRange[0],transformedRange[1],bins+1)
    return inverseLogishTransform(transformedBinEdges, T)

def logishTransform(a, linCutOff):
    tA = np.empty_like(a).astype(float)
    a_idx=0
    while a_idx < len(a):
        if a[a_idx] >= linCutOff:
           tA[a_idx] = np.log(10 * a[a_idx] / linCutOff)/np.log(10)
        else:
            tA[a_idx] = (a[a_idx]/linCutOff + np.log(10.0) - 1)/np.log(10)
        a_idx+=1
    return tA

def inverseLogishTransform(a, linCutOff):
    invA=np.empty_like(a).astype(float)
    a_idx=0
    while a_idx < len(a):
        if a[a_idx] >= 1.0: #transformed linCutOff, always 1.0; np.log(10 * linCutOff / linCutOff)/np.log(10) -> np.log(10)/np.log(10) = 1 
            invA[a_idx] = linCutOff*np.exp(np.log(10)*a[a_idx])/10
            #invA[a_idx]= (np.exp(a[a_idx])+10)*linCutOff/10
        else:
            invA[a_idx] = linCutOff*(np.log(10.0)*a[a_idx] - np.log(10.0) + 1)
        a_idx+=1
    return invA

def ticFormatter(x, T, vmin, vmax):
    if x<=T:
        return x 
    else:
        exp = np.log10(x)
        return "$10^{%d}$" % int(exp)

def addAxLine(fig, ax, pos, orientation, size=2, scale='linear', T=1000):
    if not all(i in ['linear', 'logish'] for i in [scale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logish'")
    if orientation.lower()=='vertical':
        if scale=='logish':
            lims=ax.get_xlim()
            vmin = lims[0]
            vmax = lims[1]
            pos = convertToLogishPlotCoordinates([pos],vmin,vmax,T)
        ax.axvline(pos, c='r')
    else:
        if scale=='logish':
            lims=ax.get_ylim()
            vmin = lims[0]
            vmax = lims[1]
            pos = convertToLogishPlotCoordinates([pos],vmin,vmax,T)
        ax.axhline(pos,  c='r')
    return fig

def addLine(fig, ax, lStartCoordinate, lEndCoordinate, size=2, scale='linear', T=1000):
    if not all(i in ['linear', 'logish'] for i in [scale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logish'")
    if scale.lower()=='logish':
        view=ax.xaxis.get_view_interval()
        xCoordinates=convertToLogishPlotCoordinates([lStartCoordinate[0],lEndCoordinate[0]], vmin=view[0], vmax=view[1], T=T)
        view=ax.yaxis.get_view_interval()
        yCoordinates=convertToLogishPlotCoordinates([lStartCoordinate[1],lEndCoordinate[1]], vmin=view[0], vmax=view[1], T=T)
        lStartCoordinate=[xCoordinates[0],yCoordinates[0]]
        lEndCoordinate=[xCoordinates[1],yCoordinates[1]]
    plt.plot([lStartCoordinate[0], lEndCoordinate[0]], [lStartCoordinate[1], lEndCoordinate[1]], color='r', linestyle='-', linewidth=size,figure=fig)
    return fig, ax

def addArrow(fig, ax, lStartCoordinate, lEndCoordinate, size=5000):
    arrow=Arrow(lStartCoordinate[0],lStartCoordinate[1],lEndCoordinate[0]-lStartCoordinate[0],lEndCoordinate[1]-lStartCoordinate[1],width=size, transform=ax.transAxes,head_width=size, head_length=size, fc='r', ec='r')
    #ax.arrow(lStartCoordinate[0], lStartCoordinate[1], lEndCoordinate[0]-lStartCoordinate[0], lEndCoordinate[1]-lStartCoordinate[1], head_width=size, head_length=size, fc='r', ec='r')
    ax.add_patch(arrow)
    return fig

def draw_ellipse(position, covariance, sigma=2, ax=None, **kwargs):
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if plot:
        ax = ax or plt.gca();
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width,height = np.sqrt(s)*sigma
    else:
        angle = 0
        width, height = np.sqrt(covariance)*sigma
    #Note width, height here is the full width and height and not the semiaxis length
    # Draw the Ellipse
    if plot:
        ax.add_patch(Ellipse(position, width, height,
                                 angle, **kwargs));
    return width, height, angle

def plot_gmm(fcsDF, xCol, yCol, vI, gmm, sigma, ax):  
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False    
    ax = ax or plt.gca()
    vEllipses=[]
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_): 
        width, height, angle = draw_ellipse(pos, covar, sigma, fill=False,edgecolor='#FF0000', linestyle='dashed');
        vEllipses.append([pos,width,height,angle])
    if plot:
        plt.show();
    return vEllipses

def plot_densityFunc(fcsDF, xCol,vI=sentinel, sigma=3, bins=300, scale='linear',  T=1000, *args, **kwargs):
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n")
        return None
    if len(vI)<bins:
        sys.stderr.write("Fewer events than bins, readjusting number of bins\n")
        bins=len(vI)
    if not all(i in ['linear', 'logish'] for i in [scale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logish'")
    if not isinstance(sigma,(float,int)): 
        raise AliGaterError("Sigma must be float or int, found: "+str(type(sigma)),"in plot_densityFunc")
    if 'sigma' in kwargs:
            if not isinstance(kwargs['sigma'],(float,int)):
                raise AliGaterError("Sigma must be float or int, found: "+str(type(sigma)),"in plot_densityFunc")
            else:
                sigma=kwargs['sigma']
    if 'bins' in kwargs:
        if not isinstance(kwargs['bins'],int):
            raise AliGaterError("sigma must be int, found: "+str(type(bins)),"in plot_densityFunc")
        else:
            bins=kwargs['bins']
    data=getGatedVector(fcsDF, xCol, vI, return_type="nparray")
    if scale == 'logish':
        BinEdges=logishBin(data,bins,T)
        histo = np.histogram(data, BinEdges)
    else:
        histo=np.histogram(data, bins)
    vHisto=np.linspace(min(histo[1]),max(histo[1]),bins)
    smoothedHisto=gaussian_filter1d(histo[0].astype(float),sigma)
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(vHisto,smoothedHisto, label="pdf for "+str(xCol)+", sigma: "+str(sigma))
    plt.legend()
    if scale.lower()=='logish':
        ax=plt.gca()
        ax.set_xlim(left=min(data),right=max(data))
        ax.xaxis.set_major_locator(LogishLocator())
        ax.xaxis.set_major_formatter(LogishFormatter())
    #plt.show()
    return fig,ax

from sklearn.decomposition import PCA

def imagePCA_cluster(imlist, samplelist, nOfComponents=2):
    immatrix = np.array([im.flatten() for im in imlist],'f')
    if immatrix.shape[0] == 0:
        reportStr="ERROR: imagedata missing\n"
        sys.stderr.write(reportStr)
        return #TODO: SHOULD BE RAISE
    if immatrix.shape[0] < nOfComponents:
        reportStr="WARNING: fewer samples than requested components for PC analysis, adjusting\n"
        sys.stderr.write(reportStr)
        nOfComponents=immatrix.shape[0]
    pca_obj = PCA(n_components=nOfComponents)
    pca_obj.fit(immatrix)
    projection_d = pca_obj.transform(immatrix)
    projection_d_df = pd.DataFrame(projection_d)
    projection_d_df.index = samplelist
    columnNames=[]
    for i in np.arange(1,nOfComponents+1,1):
        columnStr="PC"+str(i)
        columnNames.append(columnStr)
    projection_d_df.columns = columnNames
    reportStr="PCs explained variance: \n"+str(pca_obj.explained_variance_ratio_)+"\n"
    sys.stderr.write(reportStr)
    #center the coordinate system on the mean of each PC
    projection_d_df = projection_d_df - projection_d_df.mean()
    if nOfComponents==2:
        axes = projection_d_df.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))
        axes.set_xlim([projection_d_df['PC2'].min()*1.1, projection_d_df['PC2'].max()*1.1])
        axes.set_ylim([projection_d_df['PC1'].min()*1.1, projection_d_df['PC1'].max()*1.1])
        plt.show()
    projection_d_df['length']=np.sqrt(np.square(projection_d_df).sum(axis=1))
    projection_d_df.sort_values(by='length', inplace=True)
    return projection_d_df
    
def imagePCA(imlist):
    """
    Principal Component Analysis of heatmap images, plots results.
    Takes a list-like of heatmap images, flattens them and calls image_pca.
    
    **Parameters**
    
    X, list-like
        List-like matrix with training data stored as flattened arrays in rows.
        
    **Returns** 
   
    None

    **Examples**

    None currently.
    """
    m=16
    n=16
    # create matrix to store all flattened images
    immatrix = np.array([im.flatten() for im in imlist],'f')
    
    # perform PCA
    V, S, immean = image_pca(immatrix)
    #show some images (mean and 7 first modes)
    plt.figure(figsize=(20,30))
    plt.subplot(2,4,1)
    plt.imshow(immean.reshape(m,n))
    for i in range(20):
      plt.subplot(4,5,i+1)
      plt.imshow(V[i].reshape(m,n).T[::-1], cmap="bwr")
    plt.show()


def image_pca(X):
    """
    Principal Component Analysis of heatmap images, main purpose is to be called internally by imagePCA
    
    **Parameters**
    
    X, list-like
        List-like matrix with training data stored as flattened arrays in rows.
        
   **Returns** 
   
    List-like    
        Projection matrix (with important dimensions first)
    Float
        Variance
    Float
        Mean

    **Examples**

    None currently.
    """

    # get dimensions
    #TODO: sloppy error handling
    try:
        num_data,dim = X.shape
    except ValueError:
        sys.stderr.write("WARNING, in image_pca: input matrix seems invalid\n")
        return None,None,None
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim>num_data:
        # PCA - compact trick used
        M = np.dot(X,X.T) # covariance matrix
        e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = np.dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = np.linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V,S,mean_X

def is_decade(x, base=10):
    if not np.isfinite(x):
        return False
    if x == 0.0:
        return True
    lx = np.log(np.abs(x)) / np.log(base)
    return is_close_to_int(lx)

def is_close_to_int(x):
    if not np.isfinite(x):
        return False
    return abs(x - nearest_int(x)) < 1e-10

def nearest_int(x):
    if x == 0:
        return int(0)
    elif x > 0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)

        
def convertToLogishPlotCoordinates(Ticlocs, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    tTiclocs=logishTransform(Ticlocs, T)
    plotTics=[]
    for tTic in tTiclocs:
        plotTic=(tTic-tMinMax[0])/transformedRange*actualRange+vmin
        plotTics.append(plotTic)
    assert len(tTiclocs)==len(Ticlocs)
    return plotTics

def convertToLogishPlotCoordinate(Ticloc, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    tTicloc=logishTransform([Ticloc], T)[0]
    plotTic=(tTicloc-tMinMax[0])/transformedRange*actualRange+vmin
    return plotTic

def invertLogishPlotcoordinates(plotTics, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    invPlotTics=[]
    for tTic in plotTics:
        invPlotTic=(tTic-vmin)/actualRange*transformedRange+tMinMax[0]
        invPlotTics.append(invPlotTic)
    result=inverseLogishTransform(invPlotTics, T)
    return result

def invertLogishPlotcoordinate(plotTic, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    invPlotTic=(plotTic-vmin)/actualRange*transformedRange+tMinMax[0]
    result=inverseLogishTransform([invPlotTic], T)[0]
    return result



class LogishLocator(Locator):
    """
    Determine the tick locations for logish axes based on LogLocator
    Hacked version of LogLogator that covers normal usecases of the logish scale
    Only defined with ticlocations for data in range -50000 < x < 1 000 000
    """

    def __init__(self, linCutOff=1000, subs=(1.0,), numdecs=4, numticks=None):
        """
        Place ticks on the locations : subs[j] * base**i
        Parameters
        ----------
        subs : None, string, or sequence of float, optional, default (1.0,)
            Gives the multiples of integer powers of the base at which
            to place ticks.  The default places ticks only at
            integer powers of the base.
            The permitted string values are ``'auto'`` and ``'all'``,
            both of which use an algorithm based on the axis view
            limits to determine whether and how to put ticks between
            integer powers of the base.  With ``'auto'``, ticks are
            placed only between integer powers; with ``'all'``, the
            integer powers are included.  A value of None is
            equivalent to ``'auto'``.
        """
        if numticks is None:
            if rcParams['_internal.classic_mode']:
                numticks = 15
            else:
                numticks = 'auto'
                
        self._base=np.exp(1)
        self.subs(subs)
        self.numdecs = numdecs
        self.numticks = numticks
        self.T = linCutOff
        
    def set_params(self, subs=None, numdecs=4, numticks=None):
        """Set parameters within this locator."""
        if subs is not None:
            self.subs(subs)
        if numdecs is not None:
            self.numdecs = numdecs
        if numticks is not None:
            self.numticks = numticks

    # FIXME: these base and subs functions are contrary to our
    # usual and desired API.

    def subs(self, subs):
        """
        set the minor ticks for the log scaling every base**i*subs[j]
        """
        if subs is None:  # consistency with previous bad API
            self._subs = 'auto'
        elif isinstance(subs, six.string_types):
            if subs not in ('all', 'auto'):
                raise ValueError("A subs string must be 'all' or 'auto'; "
                                 "found '%s'." % subs)
            self._subs = subs
        else:
            self._subs = np.asarray(subs, dtype=float)

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.view_limits()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if self.numticks == 'auto':
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks
        
    #if vmin < self.T:
        tmpTicLoc=[-50000, -40000, -30000, -20000, -10000, -5000,-4000,-3000, -2000, -1000, 0]
        Ticlocs = list(set(np.clip(tmpTicLoc, vmin, self.T)))
        Ticlocs = list(np.sort(Ticlocs))
    
    #if vmax > self.T:
        tmpTicLoc=[0,1000]
        tmpTicLoc.extend(np.arange(1000.0,10001,1000))
        tmpTicLoc.extend(np.arange(10000.0,100001,10000))
        tmpTicLoc.extend(np.arange(100000.0,1000000,100000)) #[10000.0,100000.0, 200000, 300000, 1000000.0]
        Ticlocs.extend(tmpTicLoc)
        clip_Ticlocs=list(set(np.clip(Ticlocs,vmin, vmax)))
        Ticlocs=np.sort(clip_Ticlocs)
        #ADD HOC POSSIBLY
        Ticlocs=Ticlocs[1:-1]
        Ticlocs=convertToLogishPlotCoordinates(Ticlocs, vmin, vmax, self.T)
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        return self.raise_if_exceeds(np.asarray(Ticlocs))

    def view_limits(self, vmin=None, vmax=None):
        'Try to choose the view limits intelligently'
        vmin, vmax = self.axis.get_view_interval()
        return vmin, vmax


class LogishFormatter(Formatter):
    """
    Base class for formatting ticks on a logish scale.
    Hacked version of LogFormatter that covers normal usecases of the logish scale
    Only defined with formatting ticlabels for data in range -50000 < x < 1 000 000

    
    Parameters
    ----------
    labelOnlyBase : bool, optional, default: False
        If True, label ticks only at integer powers of base.
        This is normally True for major ticks and False for
        minor ticks.
    minor_thresholds : (subset, all), optional, default: (1, 0.4)
        If labelOnlyBase is False, these two numbers control
        the labeling of ticks that are not at integer powers of
        base; normally these are the minor ticks. The controlling
        parameter is the log of the axis data range.  In the typical
        case where base is 10 it is the number of decades spanned
        by the axis, so we can call it 'numdec'. If ``numdec <= all``,
        all minor ticks will be labeled.  If ``all < numdec <= subset``,
        then only a subset of minor ticks will be labeled, so as to
        avoid crowding. If ``numdec > subset`` then no minor ticks will
        be labeled.
    linthresh : float, optional, default: 1000
        The threshold for the logicle scale change from linear-like to log-like scaling
 
    Notes
    -----
    The `set_locs` method must be called to enable the subsetting
    logic controlled by the ``minor_thresholds`` parameter.
    In some cases such as the colorbar, there is no distinction between
    major and minor ticks; the tick locations might be set manually,
    or by a locator that puts ticks at integer powers of base and
    at intermediate locations.  For this situation, disable the
    minor_thresholds logic by using ``minor_thresholds=(np.inf, np.inf)``,
    so that all ticks will be labeled.
    To disable labeling of minor ticks when 'labelOnlyBase' is False,
    use ``minor_thresholds=(0, 0)``.  This is the default for the
    "classic" style.
    Examples
    --------
    To label a subset of minor ticks when the view limits span up
    to 2 decades, and all of the ticks when zoomed in to 0.5 decades
    or less, use ``minor_thresholds=(2, 0.5)``.
    To label all minor ticks when the view limits span up to 1.5
    decades, use ``minor_thresholds=(1.5, 1.5)``.
    """
    def __init__(self, labelOnlyBase=False,
                 minor_thresholds=None,
                 linthresh=1000):
        
        self.labelOnlyBase = labelOnlyBase
        if minor_thresholds is None:
            if rcParams['_internal.classic_mode']:
                minor_thresholds = (0, 0)
            else:
                minor_thresholds = (1, 0.4)
        self.minor_thresholds = minor_thresholds
        self._sublabels = None
        self._linthresh = linthresh
        self._base = np.exp(1)



    def _num_to_string(self, x, vmin, vmax):
        x = round(x,0)
        if not x in [-5000, -4000, -3000, -2000, -1000, 0 ,1000,10000,100000,1000000]:
            s = ''
        else:
            s = self.pprint_val(x, vmax - vmin)
        return s

    def __call__(self, x, pos=None):
        """
        Return the format for tick val `x`.
        """
        if x == 0.0:  # Symlog
            return '0'
        vmin, vmax = self.axis.get_view_interval()
        #tVals = logishTransform([vmin, vmax, x], self._linthresh)
        # only label the decades
        #fx = (x-vmin)/(vmax-vmin)*(tVals[1] - tVals[0])-tVals[0]
        #fx = inverseLogishTransform([fx],self._linthresh)[0]
        fx=invertLogishPlotcoordinate(x,vmin,vmax,self._linthresh)
        #print(fx)
        s = self._num_to_string(fx, vmin, vmax)
        return self.fix_minus(s)
        

    def pprint_val(self, x, d):
        #If the number is at or below the set lin-cutoff (_lintrehsh)
        #Print it as an int
        #TODO: WHY DO I NEED THE +1 HERE?
        if x <= self._linthresh+1:
            return '%d' % x

        fmt = '%1.3e'
        s = fmt % x
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            if exponent:
                if float(mantissa) > 1:
                    s = '$%s*10^{%d}$' % (mantissa, exponent)
                else:
                    s = '$%s0^{%d}$' % (mantissa, exponent)
            else:
                s = mantissa
        else:
            s = s.rstrip('0').rstrip('.')
        return s
