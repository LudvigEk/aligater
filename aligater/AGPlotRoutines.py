#	______|\__________\o/__________
#			~~aliGater~~
#	(semi)automated gating software
#
#	Utilizing Dislin for plots, non-commercial use only!
#	Relevant EULA section:
#	Grant of Free Usage
#	You are allowed to use DISLIN for free as a private person or as a member of an institute that does not earn money with selling any products and services.
#	http://dislin.de
#
#	Utilizing Intels Math Kernel Library (MKL)
#
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	http://nilssonlab.org
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Ellipse
import aligater as ag
import matplotlib.ticker as ticker
#from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d#, gaussian_filter
import sys

sentinel = object()
def plotHeatmap(fcsDF, x, y, vI=sentinel, bins=300, scale='linear', xscale='linear', yscale='linear', thresh=1000):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events")
        return None
    if len(vI)<bins:
        bins=len(vI)
    vX=ag.getGatedVector(fcsDF, x, vI)
    vY=ag.getGatedVector(fcsDF, y, vI)
    plt.clf()
    matplotlib.rcParams['image.cmap'] = 'jet'
    heatmap, xedges, yedges = getHeatmap(vX, vY, bins, scale, xscale, yscale)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap=np.ma.masked_where(heatmap == 0, heatmap)
    
    plt.clf()
    fig, ax = plt.subplots()
    plt.imshow(heatmap.T, extent=extent, origin='lower',aspect='auto')
    plt.xlabel(x)
    plt.ylabel(y)
    cmap=plt.get_cmap()
    cmap.set_bad(color='white') #Zeroes should be white, not blue
    if scale.lower()=='logish':
        ticks = ticker.FuncFormatter(lambda x, pos: '{0}'.format(ticFormatter(x, thresh)))
        ax.set_xscale('logish', linCutOff=thresh)
        ax.xaxis.set_major_formatter(ticks)
    return fig,ax

def getHeatmap(vX, vY, bins, scale, xscale, yscale, T=1000):
    if not any(isinstance(i,str) for i in [scale,xscale,yscale]):
        raise TypeError("scale, xscale, yscale must be specified as string, such as: 'linear', 'log'")
    if not all(i in ['linear', 'logish'] for i in [scale,xscale,yscale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'log'")
    if scale=='linear' and xscale=='linear' and yscale == 'linear':
        return np.histogram2d(vX, vY, bins)
    elif scale=='logish':
        xBinEdges=logishBin(vX,bins,T)
        yBinEdges=logishBin(vY,bins,T)
        #SILVERTEJPLÖSNING
        for index,elem in enumerate(np.diff(yBinEdges)):
            if elem <=0:
                 print(index)
                 raise
        for index,elem in enumerate(np.diff(xBinEdges)):
            if elem <=0:
                 print(index)
                 raise
        #print(np.diff(yBinEdges))
        #return np.histogram2d(vX, vY, bins)
        return np.histogram2d(vX, vY, [xBinEdges,yBinEdges])

def logishBin(vX, bins, T):
    defaultRange=[min(vX),max(vX)]
    transformedRange=logishTransform(defaultRange,T)
    transformedBinEdges=np.linspace(transformedRange[0],transformedRange[1],bins+1)
    return inverseLogishTransform(transformedBinEdges, T)

def logishTransform(a, linCutOff):
    tA = np.empty_like(a)
    a_idx=0
    while a_idx < len(a):
        if a[a_idx] >= linCutOff:
           tA[a_idx] = np.log(10 * a[a_idx] / linCutOff)/np.log(10)
        else:
            tA[a_idx] = (a[a_idx]/linCutOff + np.log(10.0) - 1)/np.log(10)
        a_idx+=1
    return tA

def inverseLogishTransform(a, linCutOff):
    invA=np.empty_like(a)
    a_idx=0
    while a_idx < len(a):
        if a[a_idx] <= 1.0: #transformed linCutOff, always 1.0; np.log(10 * linCutOff / linCutOff)/np.log(10) -> np.log(10)/np.log(10) = 1 
             invA[a_idx] = linCutOff*(np.log(10.0)*a[a_idx] - np.log(10.0) + 1)
        else:
            invA[a_idx]= (np.exp(a[a_idx])+10)*linCutOff/10
        a_idx+=1
    return invA

def ticFormatter(x, T):
    if x<=T:
        return x 
    else:
        exp = x // 1000
        return "$10^{%d}$" % int(exp)

def addLine(fig, ax, lStartCoordinate, lEndCoordinate, size=2):
    ax.plot([lStartCoordinate[0], lEndCoordinate[0]], [lStartCoordinate[1], lEndCoordinate[1]], color='r', linestyle='-', linewidth=size)
    return fig

def addArrow(fig, ax, lStartCoordinate, lEndCoordinate, size=5000):
    ax.arrow(lStartCoordinate[0], lStartCoordinate[1], lEndCoordinate[0]-lStartCoordinate[0], lEndCoordinate[1]-lStartCoordinate[1], head_width=size, head_length=size, fc='r', ec='r')
    return fig

def draw_ellipse(position, covariance, sigma=2, ax=None, **kwargs):
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
    ax.add_patch(Ellipse(position, width, height,
                             angle, **kwargs));
    return width, height, angle

def plot_gmm(fcsDF, xCol, yCol, vI, gmm, sigma, ax):  
    ax = ax or plt.gca()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
       width, height, angle = draw_ellipse(pos, covar, sigma ,fill=False,edgecolor='#FF0000', linestyle='dashed');
    plt.show();
    return pos, width, height, angle

def plot_densityFunc(fcsDF, xCol,vI=sentinel, sigma=3, bins=300):
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")

    data=ag.getGatedVector(fcsDF, xCol, vI, return_type="nparray")
    histo=np.histogram(data, bins)
    vHisto=np.linspace(min(histo[1]),max(histo[1]),bins)
    smoothedHisto=gaussian_filter1d(histo[0],sigma)
    plt.clf()
    plt.plot(vHisto,smoothedHisto, label="pdf for "+str(xCol)+", sigma: "+str(sigma))
    plt.legend()
    plt.show()
    return None

def main():
	return None

if __name__ == '__main__':
	main()