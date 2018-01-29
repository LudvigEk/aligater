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

def gmm2D(fcsDF, gate1, gate2, nOfComponents, vI=sentinel):
    if vI is sentinel:
        vI=fcsDF.index
    #randomly_sampled = np.random.choice(vI, size=len(vI), replace=False)
    vX = getGatedVector(fcsDF,gate1,vI=vI,return_type="nparray")
    vY = getGatedVector(fcsDF,gate2,vI=vI,return_type="nparray")
    fcsArray=np.array([vX,vY]).T
    gmm = GaussianMixture(n_components=nOfComponents)
    gmm.fit(fcsArray)
    return gmm

def gateGMM(fcsDF,xCol, yCol, gmm, vI=sentinel, sigma=1, plot=True):
    if vI is sentinel:
        vI=fcsDF.index
    if not isinstance(gmm, GaussianMixture):
        raise TypeError("gmm argument must be a sklearn.mixture GaussianMixture object")
    vOutput=[]
    fig,ax = ag.plotHeatmap(fcsDF, xCol, yCol, vI, aspect='equal')
    pos, width, height, angle = ag.plot_gmm(fcsDF,xCol, yCol, vI, gmm, sigma, ax)
    #plot_ggm will return the full width and height of the drawn ellipse, whereas
    #gateEllipsoid wants the length of the semiaxis
    vOutput = ag.gateEllipsoid(fcsDF, xCol, yCol, pos[0], pos[1], width/2, height/2, np.radians(angle), vI)
    fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol, vOutput)
    return vOutput
    
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
    if return_type.lower() not in ["pdseries","nparray"]:
        raise TypeError("Specify return type as 'pdseries' or 'nparray'")
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
    
def gateThreshold(fcsDF, xCol, yCol, thresh, orientation="horisontal", vI=sentinel, population="upper", plot=True):
    if vI is sentinel:
        vI=fcsDF.index
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if population.lower() not in ["upper","lower"]:
        raise TypeError("Specify desired population, 'upper' or 'lower' in regard to set threshold")
    if orientation.lower() not in ["horisontal","vertical"]:
        raise TypeError("Specify desired population, 'upper' or 'lower' in regard to set threshold") 
        
    if population.lower() == "upper":    
        if orientation.lower() == "horisontal":
            vOutput=fcsDF[fcsDF[xCol]>thresh].index
        else:
            vOutput=fcsDF[fcsDF[yCol]>thresh].index
    else:
        if orientation.lower() == "horisontal":
            vOutput=fcsDF[fcsDF[xCol]<thresh].index
        else:
            vOutput=fcsDF[fcsDF[yCol]<thresh].index
    print(len(vOutput))
    if plot:
        fig,ax = ag.plotHeatmap(fcsDF, xCol, yCol, vI)
        ylim=ax.get_ylim()
        xlim=ax.get_ylim()
        if orientation.lower() == "horisontal":
            ag.addLine(fig,ax,[thresh,ylim[0]],[thresh,ylim[1]])
        else:   #Vertical
            ag.addLine(fig,ax,[xlim[0],thresh],[xlim[1],thresh])
        ag.plt.show()
        ag.plotHeatmap(fcsDF, xCol, yCol, vOutput)
        ag.plt.show()
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
        
    #Faster alternatives than zipping like below? pd.iterrows() is definetly slower
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
        meanY=centerCoord[1]
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

def getPCSemiAxis(center, eigen1, eigen2, eigen1Scale=1, eigen2Scale=1):
    if not all(isinstance(i, list) for i in [center, eigen1, eigen2]):
        raise TypeError("Input arguments for getPrincipalComponentsemiAxis (barycenter, eigen1, eigen2) must be list.")

    majorAxis = np.sqrt(eigen1[0])*eigen1Scale
    minorAxis = np.sqrt(eigen2[0])*eigen2Scale
    
    eigen1X=majorAxis*eigen1[1]+center[0]
    eigen1Y=majorAxis*eigen1[2]+center[1]
    
    eigen2X=minorAxis*eigen2[1]+center[0]
    eigen2Y=minorAxis*eigen2[2]+center[1]  
    
    PC1=[eigen1X, eigen1Y]
    PC2=[eigen2X, eigen2Y]

    return center, PC1, PC2

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
    #Re-estimate original index, note +- range/bins error
    xCoord=highestPoint[0]*(xmax-xmin)/bins + xmin
    yCoord=highestPoint[1]*(ymax-ymin)/bins + ymin
    return [xCoord, yCoord]
    
def reportGateResults(vI, vOutput):
    reportString="After gating, "+str(len(vOutput))+" out of "+str(len(vI))+" events remain."
    sys.stderr.write(reportString)

def gatePC(fcsDF, xCol, yCol, vI=sentinel, widthScale=1, heightScale=1, center='centroid', customCenter=None, plot=True):
    if vI is sentinel:
        vI=fcsDF.index
    if (xCol not in fcsDF.columns or yCol not in fcsDF.columns):
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if center.lower() not in ['centroid','density','custom']:
        raise ValueError("Specify center/anchor point for PC analysis; centroid, density or custom")
    elif center.lower() == 'custom' and type(customCenter) is not list:
        print(type(customCenter))
        raise TypeError("If custom center is specified the 'customCenter' argument must be passed as a list of two, i.e. [x,y]")
    elif center.lower() == 'custom' and type(customCenter) is not list:
        if len(customCenter) != 2:
            raise TypeError("If custom center is specified the 'customCenter' argument must be passed as a list of two, i.e. [x,y]")
    if type(plot) is not type(True):
        raise TypeError("Plot argument should be specified as bool (True/False)")
        
    if center.lower() == 'density':
        center=ag.getHighestDensityPoint(fcsDF, xCol, yCol, vI)
    elif center.lower() == 'centroid':
        center=None
    else:
        center=customCenter
        
    center, eigen1, eigen2 = ag.getPCs(fcsDF, xCol, yCol, center, vI)
    center, PC1, PC2 = ag.getPCSemiAxis(center, eigen1, eigen2, widthScale, heightScale)
    width=ag.getVectorLength(center, PC1)
    height=ag.getVectorLength(center, PC2)
    angle=ag.calculateAngle(center, PC1)
    result=ag.gateEllipsoid(fcsDF, xCol, yCol,xCenter=center[0],yCenter=center[1],majorRadii=width, minorRadii=height,theta=angle,vI=vI)
    if plot:
        fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')
        ag.addLine(fig, ax, center, PC1)
        ag.addLine(fig, ax, center, PC2)
        ax.add_patch(ag.Ellipse(center, 2*width, 2*height, np.degrees(angle),fill=False,edgecolor='#FF0000', linestyle='dashed'))
        ag.plt.show()
        ag.plotHeatmap(fcsDF, xCol, yCol,result)
        ag.plt.show()
    return result

def getVectorCoordiantes(length, angle):
    theta=math.degrees(angle)
    y = length*math.asin(theta)
    x = length*math.acos(theta)
    return[x,y]

def getDensityFunc(fcsDF, xCol,vI=sentinel, sigma=3, bins=300):
    data=ag.getGatedVector(fcsDF, xCol, vI, return_type="nparray")
    histo=np.histogram(data, bins)
    smoothedHisto=ag.gaussian_filter1d(histo[0],sigma)
    #vHisto=np.linspace(min(histo[1]),max(histo[1]),bins)
    return smoothedHisto, histo[1]

def valleySeek(fcsDF, xCol, vI=sentinel, interval=['start','end'], sigma=3, bins=300):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        raise ValueError("Passed index contains no events")    
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if type(interval) is not list:
        raise ValueError("Interval must be specified as list of two: [x,y].\nInterval can be half open to either side, i.e. ['start',y] or [x,'end'].")
    if len(interval)!=2:
        raise ValueError("Interval must be specified as list of two: [x,y].\nInterval can be half open to either side, i.e. ['start',y] or [x,'end'].")

    smoothedHisto, binData=getDensityFunc(fcsDF,xCol, vI, sigma, bins)
  
    if type(interval[0]) is str:
        if interval[0].lower() in ['start', 'begin','first']:
            interval[0]=min(binData) 
        else:
            raise ValueError("Interval must be specified as list of two: [x,y].\nInterval can be half open to either side by giving ['start', y] or [x ,'end'].")
    if type(interval[1]) is str:
        if interval[1].lower() in ['end', 'stop', 'last']:
            interval[1]=max(binData) 
        else:
            raise ValueError("Interval must be specified as list of two: [x,y].\nInterval can be half open to either side by giving ['start', y] or [x ,'end'].")
    if interval[1] > max(binData):
        interval[1] = max(binData)
    vIndicies=[]
    for index, x in np.ndenumerate(binData):
        #Note the non-inclusive upper bound, critical to stay in-bound of array index
        if x >= interval[0] and x < interval[1]:
            vIndicies.append(index[0])
            
    if len(vIndicies)<=3:
        raise ValueError("Specified interval is too narrow (Not enough data points to find a valley)")

    minVal=np.inf
    minValIndex=0
    for index in vIndicies:
        if smoothedHisto[index] < minVal:
            minVal=x
            minValIndex=index
    return (binData[minValIndex+1]+binData[minValIndex])/2
            
def quadGate(fcsDF, xCol, yCol, xThresh, yThresh, vI=sentinel):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        raise ValueError("Passed index contains no events") 
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if not all(isinstance(i,(float, int)) for i in [xThresh, yThresh]):
        raise TypeError("xThresh, yThresh must be specified as integer or floating-point values")
    vX, vY = ag.getGatedVectors(fcsDF, xCol, yCol, vI, return_type="nparray")
    assert(len(vX)==len(vY))
    vTopLeft=[]
    vTopRight=[]
    vBottomRight=[]
    vBottomLeft=[]
    for x,y, index in zip(vX, vY, vI):
        if x < xThresh and y > yThresh:
            vTopLeft.append(index)
        elif x > xThresh and y > yThresh:
            vTopRight.append(index)
        elif x > xThresh and y < yThresh:
            vBottomRight.append(index)
        elif x < xThresh and y < yThresh:
            vBottomLeft.append(index)
        else:
            raise RuntimeError("Unexpected error in quadGate")
    counter=0
    for event in [len(vTopLeft),len(vTopRight),len(vBottomRight),len(vBottomLeft)]:
        if event == 0:
            counter=counter+1
    if counter != 0:
        errStr=str(counter)+" quadrant(s) contain no events"
        sys.stderr.write(errStr)
    if counter==4:
        sys.stderr.write("No quadrant contains events")
        return None
    return vTopLeft, vTopRight, vBottomRight, vBottomLeft