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
#from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d#, gaussian_filter
import sys

sentinel = object()
def plotHeatmap(fcsDF, x, y, vI=sentinel, bins=300):
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
    heatmap, xedges, yedges = np.histogram2d(vX, vY, bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap=np.ma.masked_where(heatmap == 0, heatmap)
    
    plt.clf()
    fig, ax = plt.subplots()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.xlabel(x)
    plt.ylabel(y)
    cmap=plt.get_cmap()
    cmap.set_bad(color='white') #Zeroes should be white, not blue
    return fig, ax

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