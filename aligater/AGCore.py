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
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	http://nilssonlab.org

import aligater as ag
import pandas as pd
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
sentinel = object()

def gaussianMM2D(fcsDF, gate1, gate2, nOfComponents, sigma, vI=sentinel):
    randomly_sampled = np.random.choice(vI, size=len(vI), replace=False)
    vX = getGatedVector(fcsDF,gate1,vI=randomly_sampled,return_type="nparray")
    vY = getGatedVector(fcsDF,gate2,vI=randomly_sampled,return_type="nparray")
    sampledSubset=np.array([vX,vY]).T
    gmm = GaussianMixture(n_components=nOfComponents)
    gmm.covariances_ = sigma
    gmm.fit(sampledSubset)
    X = getGatedVectors(fcsDF,gate1,gate2,vI=vI,return_type="ndarray").T
    plot_gmm(gmm, X)



from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.2, s=1, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def getGatedVector(fcsDF, gate, vI=sentinel, return_type="pdseries"):
    if return_type.lower() not in ["pdseries","nparray"]:
        raise TypeError("Specify return type as 'pdseries' or 'nparray'")
    if vI is sentinel:
        vI=fcsDF.index
    if return_type.lower()=="pdseries":
        gated_vector=fcsDF[gate].loc[vI]
    else:
        gated_vector=fcsDF[gate].loc[vI].values
    return gated_vector

def getGatedVectors(fcsDF, gate1, gate2, vI=sentinel, return_type="pdseries"):
    if return_type.lower() not in ["pdseries","ndarray"]:
        raise TypeError("Specify return type as 'pdseries' or 'ndarray'")
    if vI is sentinel:
        vI=fcsDF.index
    if return_type.lower()=="pdseries":    
        gated_vector1=fcsDF[gate1].loc[vI]
        gated_vector2=fcsDF[gate2].loc[vI]
        return gated_vector1, gated_vector2
    else:
        vX=fcsDF[gate1].loc[vI].values
        vY=fcsDF[gate2].loc[vI].values
        return np.array([vX,vY])
    
def gateThreshold(fcsDF, gate, thresh, vI=sentinel, population="upper"):
    if vI is sentinel:
        vI=fcsDF.index
    if gate not in fcsDF.columns:
        raise TypeError("specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if population.lower() not in ["upper","lower"]:
        raise TypeError("Specify desired population, 'upper' or 'lower' in regard to set threshold")    
    if population.lower() == "upper":    
        vOutput=fcsDF[fcsDF[gate]>thresh].index
        reportGateResults(vI, vOutput)
        return vOutput
    else:
        vOutput=fcsDF[fcsDF[gate]<thresh].index
        reportGateResults(vI, vOutput)
        return vOutput
    
def gateEllipsoid(fcsDF, xCol, yCol, xCenter, yCenter, majorRadii, minorRadii, theta, vI=sentinel, population="inner"):  
    if population.lower() not in ["outer","inner"]:
        raise TypeError("Specify desired population, 'outer' or 'inner' in regard to ellipsoid")    
    if not all(isinstance(i, (float, int)) for i in [xCenter, yCenter, majorRadii, minorRadii, theta]):
        raise TypeError("Input arguments for gateEllipsoid (xCenter, yCenter, radii and theta) must be int or float")
    if not xCol in fcsDF.columns:
        raise NameError("xCol not in passed dataframe's columns")
    if not yCol in fcsDF.columns:
        raise NameError("yCol not in passed dataframe's columns")
    if xCol==yCol:
        raise NameError("xCol and yCol cannot be the same")
        
    if vI is sentinel:
        vI=fcsDF.index  
        
    vOutput=[]
    #theta = math.radians(theta)
    
    vX = ag.getGatedVector(fcsDF, xCol, vI, return_type='nparray')
    vY = ag.getGatedVector(fcsDF, yCol, vI, return_type='nparray')
    assert len(vX)==len(vY)
        
    #Faster alternatives than zipping like below? pd.iterrows() definetly slower
    for x, y, index in zip(vX, vY, vI):
        leftTerm = (x - xCenter)*math.cos(theta) + (y - yCenter)*math.sin(theta)
        rightTerm = (x - xCenter)*math.sin(theta) - (y - yCenter)*math.cos(theta)
        majorSquared = majorRadii*majorRadii
        minorSquared = minorRadii*minorRadii
        result = (leftTerm*leftTerm) / majorSquared + (rightTerm*rightTerm) / minorSquared
        if (population.lower() == "inner" and result  <= 1):
            vOutput.append(index)
        elif (population.lower() == "outer" and result  >= 1):
            vOutput.append(index)
            
    if (len(vOutput) == 0 and population.lower() == "inner"):
        sys.stderr.write("No events inside ellipsoid")
    if (len(vOutput) == 0 and population.lower() == "outer"):
        sys.stderr.write("No events outside ellipsoid")
    reportGateResults(vI, vOutput)
    return vOutput

def getPCs(fcsDF, xCol, yCol, centerCoord=None, vI=sentinel):
    if not xCol in fcsDF.columns:
        raise NameError("xCol not in passed dataframe's columns")
    if not yCol in fcsDF.columns:
        raise NameError("yCol not in passed dataframe's columns")
    if xCol==yCol:
        raise NameError("xCol and yCol cannot be the same")
    if vI is sentinel:
        vI=fcsDF.index
    if centerCoord is not None:
        if type(centerCoord) is not list:
            raise TypeError("centerCoord is not of type List, pass list with two coordinates")
        if len(centerCoord) != 2:
            raise ValueError("centerCoord have the wrong dimensions (list of two expected)")
        bManualCenter=True
    else:
        bManualCenter=False

    vX=getGatedVector(fcsDF, xCol, vI, "nparray")
    vY=getGatedVector(fcsDF, yCol, vI, "nparray")
    assert len(vX)==len(vY)
    
    if bManualCenter:
        meanX=centerCoord[0]
        meanY=centerCoord[0]
    else:
        meanX=np.mean(vX)
        meanY=np.mean(vY)
        
    vX=np.subtract(vX, meanX)
    vY=np.subtract(vY, meanY)
    
    #Define necessary variables
    sumX=vX.sum()
    sumY=vY.sum()
    sumXX=np.square(vX).sum()
    sumYY=np.square(vY).sum()
    sumXY=np.multiply(vX, vY).sum()
    npix=len(vI)
    
    #Barycenter/centroid
    Xbar = sumX/npix
    Ybar = sumY/npix
    
    #Variance and covariance
    varX = sumXX/npix - Xbar*Xbar
    varY = sumYY/npix - Ybar*Ybar
    covXY = sumXY/npix - Xbar*Ybar
    sumvar = varX + varY
    diffvar = varX - varY
    discriminant = diffvar*diffvar + 4*covXY*covXY
    sqrtdiscr = np.sqrt(discriminant)
    
    #Eigenvalues; n this setting it represents the variance explained by the component and thus its square root represent the std. deviation
    lambdaplus = (sumvar + sqrtdiscr)/2
    lambdaminus = (sumvar - sqrtdiscr)/2

	#eigenvectors 
    aplus = varX + covXY - lambdaminus;
    bplus = varY + covXY - lambdaminus;
    aminus = varX + covXY - lambdaplus;
    bminus = varY + covXY - lambdaplus;

    #Normalisation
    denomPlus = np.sqrt(aplus*aplus + bplus*bplus)
    denomMinus = np.sqrt(aminus*aminus + bminus*bminus)
    aParallel = aplus/denomPlus
    bParallel = bplus/denomPlus
    aNormal = aminus/denomMinus
    bNormal = bminus/denomMinus
    
    eigenvalue1 = lambdaplus
    eigenvalue2 = lambdaminus
    
    #Return retranslated barycenters
    if not bManualCenter:
        trueBarX = Xbar + meanX
        trueBarY = Ybar + meanY
    else:
        trueBarX = meanX
        trueBarY = meanY
        
    trueBarycenter=[trueBarX, trueBarY]
    eigen1=[eigenvalue1, aParallel, bParallel]
    eigen2=[eigenvalue2, aNormal, bNormal]
    
    return trueBarycenter, eigen1, eigen2

def getPCSemiAxis(barycenter, eigen1, eigen2, eigen1Scale=1, eigen2Scale=1):
    if not all(isinstance(i, list) for i in [barycenter, eigen1, eigen2]):
        raise TypeError("Input arguments for getPrincipalComponentsemiAxis (barycenter, eigen1, eigen2) must be list.")

    majorAxis = np.sqrt(eigen1[0])*eigen1Scale
    minorAxis = np.sqrt(eigen2[0])*eigen2Scale
    
    eigen1X=majorAxis*eigen1[1]+barycenter[0]
    eigen1Y=majorAxis*eigen1[2]+barycenter[1]
    
    eigen2X=minorAxis*eigen2[1]+barycenter[0]
    eigen2Y=minorAxis*eigen2[2]+barycenter[1]
    PC1=[eigen1X, eigen1Y]
    PC2=[eigen2X, eigen2Y]
    return PC1, PC2

def getVectorLength(lStartCoordinate, lEndCoordinate):
    if not all(isinstance(i, list) for i in [lStartCoordinate, lEndCoordinate]):
        raise TypeError("Input arguments for getVectorLength (lStartCoordinate, lEndCoordinate) must be list.")
    if not len(lStartCoordinate)==len(lEndCoordinate)==2:
        raise ValueError("Input arguments for getVectorLength (lStartCoordinate, lEndCoordinate) must be lists containing two elements each.")
    length=np.sqrt(np.sum(np.square(np.subtract(lEndCoordinate,lStartCoordinate))))
    return length

def calculateAngle(lStartCoordinate, lEndCoordinate):
    if not all(isinstance(i, list) for i in [lStartCoordinate, lEndCoordinate]):
        raise TypeError("Input arguments for getVectorLength (lStartCoordinate, lEndCoordinate) must be list.")
    if not len(lStartCoordinate)==len(lEndCoordinate)==2:
        raise ValueError("Input arguments for getVectorLength (lStartCoordinate, lEndCoordinate) must be lists containing two elements each.")
    angle=math.atan(np.subtract(lEndCoordinate,lStartCoordinate)[1]/np.subtract(lEndCoordinate,lStartCoordinate)[0])
    return angle

def getHighestDensityPoint(fcsDF, xCol, yCol, vI=sentinel, bins=300):
    if vI is sentinel:
        vI=fcsDF.index
    if (xCol not in fcsDF.columns or yCol not in fcsDF.columns):
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    vX=ag.getGatedVector(fcsDF, xCol, vI)
    vY=ag.getGatedVector(fcsDF, yCol, vI)
    heatmap, xedges, yedges = np.histogram2d(vX, vY, bins)
    xmax=np.amax(vX)
    xmin=np.amin(vX)
    ymax=np.amax(vY)
    ymin=np.amin(vY)
    highestPoint=np.unravel_index(heatmap.argmax(), heatmap.shape)
    #Re-estimate original index, note +- 1/bins error
    xCoord=highestPoint[0]*(xmax-xmin)/bins + xmin
    yCoord=highestPoint[0]*(ymax-ymin)/bins + ymin
    return [xCoord, yCoord]
    
def reportGateResults(vI, vOutput):
    reportString="After gating, "+str(len(vOutput))+" out of "+str(len(vI))+" events remain."
    sys.stderr.write(reportString)

