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
from matplotlib.patches import Ellipse, Arrow
import matplotlib.lines as lines
import aligater as ag
import matplotlib.ticker as ticker
from aligater.AGClasses import LogishLocator, LogishFormatter
#from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter1d#, gaussian_filter
import sys

sentinel = object()
def plotHeatmap(fcsDF, x, y, vI=sentinel, bins=300, scale='linear', xscale='linear', yscale='linear', thresh=1000, aspect='auto'):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)<2:
        sys.stderr.write("Passed index contains no events\n")
        return None
    if len(vI)<bins:
        bins=len(vI)
    vX=ag.getGatedVector(fcsDF, x, vI)
    vY=ag.getGatedVector(fcsDF, y, vI)
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
    if scale.lower()=='logish':
        xscale='logish'
        yscale='logish'
    if xscale.lower()=='logish':
        ax=plt.gca()
        ax.xaxis.set_major_locator(LogishLocator(subs='all'))
        ax.xaxis.set_major_formatter(LogishFormatter())
    if yscale.lower()=='logish':
        ax=plt.gca()
        ax.yaxis.set_major_locator(LogishLocator(subs='all'))
        ax.yaxis.set_major_formatter(LogishFormatter())
    return fig,ax

def getHeatmap(vX, vY, bins, scale, xscale, yscale, T=1000, normalize=False, xlim=None, ylim=None):
    if not any(isinstance(i,str) for i in [scale,xscale,yscale]):
        raise TypeError("scale, xscale, yscale must be specified as string, such as: 'linear', 'logish'")
    if not all(i in ['linear', 'logish'] for i in [scale,xscale,yscale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logish'")
    assert len(vX) == len(vY)
    index_mask=[]
    for i in np.arange(len(vX)-1,-1,-1):
        if i < 0:
            print("KASS")
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
        return np.histogram2d(vX, vY, bins, normed=normalize)
    elif scale=='logish' or (xscale == 'logish' and yscale == 'logish'):
        xBinEdges=logishBin(vX,bins,T)
        yBinEdges=logishBin(vY,bins,T)
        return np.histogram2d(vX, vY, [xBinEdges,yBinEdges])
    if xscale=='logish':
        xBinEdges=logishBin(vX,bins,T)
        return np.histogram2d(vX, vY, [xBinEdges,bins])
    if yscale=='logish':
        yBinEdges=logishBin(vY,bins,T)
        return np.histogram2d(vX, vY, [bins,yBinEdges])

def logishBin(vX, bins, T):
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
            pos = ag.convertToLogishPlotCoordinates([pos],vmin,vmax,T)
        ax.axvline(pos, c='r')
    else:
        if scale=='logish':
            lims=ax.get_ylim()
            vmin = lims[0]
            vmax = lims[1]
            pos = ag.convertToLogishPlotCoordinates([pos],vmin,vmax,T)
        ax.axhline(pos,  c='r')
    return fig

def addLine(fig, ax, lStartCoordinate, lEndCoordinate, size=2, scale='linear', T=1000):
    if not all(i in ['linear', 'logish'] for i in [scale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logish'")
    if scale.lower()=='logish':
        view=ax.xaxis.get_view_interval()
        xCoordinates=ag.convertToLogishPlotCoordinates([lStartCoordinate[0],lEndCoordinate[0]], vmin=view[0], vmax=view[1], T=T)
        view=ax.yaxis.get_view_interval()
        yCoordinates=ag.convertToLogishPlotCoordinates([lStartCoordinate[1],lEndCoordinate[1]], vmin=view[0], vmax=view[1], T=T)
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
    if ag.execMode in ["jupyter","ipython"]:
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
    if ag.execMode in ["jupyter","ipython"]:
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

def plot_densityFunc(fcsDF, xCol,vI=sentinel, sigma=3, bins=300, scale='linear',  T=1000):
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

    data=ag.getGatedVector(fcsDF, xCol, vI, return_type="nparray")
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
  """  Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance
    and mean."""

  # get dimensions
  num_data,dim = X.shape

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


def main():
	return None

if __name__ == '__main__':
	main()