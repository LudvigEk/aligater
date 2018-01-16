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

sentinel = object()
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

def getGatedVectors(fcsDF, gate1, gate2, vI=sentinel):
    if vI is sentinel:
        vI=fcsDF.index
    gated_vector1=fcsDF[gate1].loc[vI]
    gated_vector2=fcsDF[gate2].loc[vI]
    return gated_vector1, gated_vector2
    
def gateThreshold(fcsDF, gate, thresh, vI=sentinel, population="upper"):
    if vI is sentinel:
        vI=fcsDF.index
    if gate not in fcsDF.columns:
        raise TypeError("specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if population.lower() not in ["upper","lower"]:
        raise TypeError("Specify desired population, 'upper' or 'lower' in regard to set threshold")
    if population.lower() == "upper":    
        vI=fcsDF[fcsDF[gate]>thresh].index
        return vI
    else:
        vI=fcsDF[fcsDF[gate]<thresh].index
        return vI
    
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
        
    tmpDF = fcsDF.iloc[vI]
    vOutput=[]
    #theta = math.radians(theta)
    
    for index, row in tmpDF.iterrows():
        x = row[xCol]
        y = row[yCol]
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
        
    return vOutput

def getPCs(fcsDF, xCol, yCol, vI=sentinel):
    if not xCol in fcsDF.columns:
        raise NameError("xCol not in passed dataframe's columns")
    if not yCol in fcsDF.columns:
        raise NameError("yCol not in passed dataframe's columns")
    if xCol==yCol:
        raise NameError("xCol and yCol cannot be the same")
    if vI is sentinel:
        vI=fcsDF.index
    
    #Collect data as np arrays, should always give the same size unless dataset corrupted
    vX=getGatedVector(fcsDF, xCol, vI, "nparray")
    vY=getGatedVector(fcsDF, yCol, vI, "nparray")
    assert len(vX)==len(vY)
    
    #Collect means and translate dataset to zero origin
    meanX=vX.mean()
    meanY=vY.mean()
    vX = np.subtract(vX, meanX)
    vY = np.subtract(vY, meanY)
    
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
    trueBarX = Xbar + meanX
    trueBarY = Ybar + meanY
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