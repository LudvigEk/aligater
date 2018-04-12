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
from scipy.ndimage.filters import gaussian_filter

def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'

def gmm2D(fcsDF, gate1, gate2, nOfComponents, vI=sentinel, *args, **kwargs):
    if vI is sentinel:
        vI=fcsDF.index
    #randomly_sampled = np.random.choice(vI, size=len(vI), replace=False)
    vX = getGatedVector(fcsDF,gate1,vI=vI,return_type="nparray")
    vY = getGatedVector(fcsDF,gate2,vI=vI,return_type="nparray")
    fcsArray=np.array([vX,vY]).T
    gmm = GaussianMixture(n_components=nOfComponents,*args,**kwargs)
    gmm.fit(fcsArray)
    return gmm

def gateGMM(fcsDF,xCol, yCol, gmm, vI=sentinel, sigma=1):
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if vI is sentinel:
        vI=fcsDF.index
    if not isinstance(gmm, GaussianMixture):
        raise TypeError("gmm argument must be a sklearn.mixture GaussianMixture object")
    if len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return []
    vOutput=[]
    if plot:
        fig,ax = ag.plotHeatmap(fcsDF, xCol, yCol, vI, aspect='equal')
    else:
        ax=None
    vEllipses = ag.plot_gmm(fcsDF,xCol, yCol, vI, gmm, sigma, ax)
    #plot_ggm will return the full width and height of the drawn ellipse, whereas
    #gateEllipsoid wants the length of the semiaxis
    #totalEvent=0
    for ellipses in vEllipses:
        xCenter=ellipses[0][0]
        yCenter=ellipses[0][1]
        width=ellipses[1]
        height=ellipses[2]
        angle=ellipses[3]
        vTmp = ag.gateEllipsoid(fcsDF, xCol, yCol, xCenter, yCenter, width/2, height/2, np.radians(angle), vI, info=False)
        vOutput.extend(vTmp)
        #totalEvent+=len(vTmp)
    vResult=list(set(vOutput))
    #print("all ellipses: "+str(totalEvent)+" the set: "+str(len(vResult)))
    ag.reportGateResults(vI, vResult)
    if plot:
        fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol, vResult)
    return vOutput
    
def getGatedVector(fcsDF, gate, vI=sentinel, return_type="pdseries", dtype=np.float64):
    if return_type.lower() not in ["pdseries","nparray"]:
        raise TypeError("Specify return type as 'pdseries' or 'nparray'")
    if return_type.lower()=="pdseries" and not (dtype is np.float64):
        sys.stderr.write("dtype specification not supported for pdseries return type, returning default dtype")
    if vI is sentinel:
        vI=fcsDF.index
    if return_type.lower()=="pdseries":
        gated_vector=fcsDF[gate].loc[vI]
    else:
        gated_vector=fcsDF[gate].loc[vI].values.astype(dtype)
    return gated_vector

def getGatedVectors(fcsDF, gate1, gate2, vI=sentinel, return_type="pdseries"):
    if return_type.lower() not in ["pdseries","nparray","matrix"]:
        raise TypeError("Specify return type as 'pdseries', 'nparray', 'matrix'")
    if vI is sentinel:
        vI=fcsDF.index
    if return_type.lower()=="pdseries":    
        gated_vector1=fcsDF[gate1].loc[vI]
        gated_vector2=fcsDF[gate2].loc[vI]
        return gated_vector1, gated_vector2
    elif return_type.lower()=="matrix":
        vX=fcsDF[gate1].loc[vI].values
        vY=fcsDF[gate2].loc[vI].values
        return np.array(list(zip(vX,vY)))
    else:
        vX=fcsDF[gate1].loc[vI].values
        vY=fcsDF[gate2].loc[vI].values
        return np.array([vX, vY])
    
#def gateThreshold(fcsDF, xCol, yCol=None, thresh=None, orientation="vertical", vI=sentinel, population="upper", scale='linear', linCutOff=1000, info=True):
#    if ag.execMode in ["jupyter","ipython"]:
#        plot=True
#    else:
#        plot=False
#    if vI is sentinel:
#        vI=fcsDF.index
#    if not isinstance(thresh, (float,int)):
#        raise TypeError("thresh must be specified as float or integer value")
#    if yCol is not None:
#        if yCol not in fcsDF.columns:
#            raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
#    if xCol not in fcsDF.columns:
#        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
#    if population.lower() not in ["upper","lower"]:
#        raise TypeError("Specify desired population, 'upper' or 'lower' in regard to set threshold")
#    if orientation.lower() not in ["horisontal","vertical"]:
#        raise TypeError("Specify desired population, 'upper' or 'lower' in regard to set threshold") 
#    if len(vI)==0:
#        sys.stderr.write("Passed index contains no events\n") 
#        return []
#    if thresh is None:
#        sys.stderr.write("Thresh passed as NoneType, assuming too few elements")
#        return []
#    if yCol is None:
#        densityPlot=True
#    else:
#        densityPlot=False
#    vOutput=[]
#    if orientation.lower() == "vertical":
#        data=ag.getGatedVector(fcsDF, xCol, vI, return_type="nparray")
#        if population.lower()=="upper":
#            for index, value in zip(vI, data):
#                if value >= thresh:
#                    vOutput.append(index)
#        else:
#            for index, value in zip(vI, data):
#                if value < thresh:
#                    vOutput.append(index)
#    if yCol is not None:
#        if orientation.lower() == "horisontal":
#            data=ag.getGatedVector(fcsDF, yCol, vI, return_type="nparray")
#            if population.lower()=="upper":
#                for index, value in zip(vI, data):
#                    if value >= thresh:
#                        vOutput.append(index)
#            else:
#                for index, value in zip(vI, data):
#                    if value < thresh:
#                        vOutput.append(index)
#
#    if plot and not densityPlot and info:
#        fig,ax = ag.plotHeatmap(fcsDF, xCol, yCol, vI, scale=scale,thresh=linCutOff)
#        ag.addAxLine(fig,ax,thresh,orientation,scale=scale, T=linCutOff)
#        plt.show()
#        plt.clf()
#        ag.plotHeatmap(fcsDF, xCol, yCol, vOutput, scale=scale)
#        plt.show()
#    if plot and densityPlot and info:
#        fig,ax =ag.plot_densityFunc(fcsDF,xCol, vI, scale=scale)
#        ag.addAxLine(fig,ax,thresh,orientation,scale=scale, T=linCutOff)
#        plt.show()
#    if info:
#        reportGateResults(vI, vOutput)
#    return vOutput
    

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
    reportString="After gating, "+str(len(vOutput))+" out of "+str(len(vI))+" events remain.\n"
    sys.stderr.write(reportString)


def gatePC(fcsDF, xCol, yCol, vI=sentinel, widthScale=1, heightScale=1, center='centroid', customCenter=None):
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False

    if vI is sentinel:
        vI=fcsDF.index
    if (xCol not in fcsDF.columns or yCol not in fcsDF.columns):
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if center.lower() not in ['centroid','density','custom']:
        raise ValueError("Specify center/anchor point for PC analysis; centroid, density or custom")
    elif center.lower() == 'custom' and type(customCenter) is not list:
        raise TypeError("If custom center is specified the 'customCenter' argument must be passed as a list of two, i.e. [x,y]")
    elif center.lower() == 'custom' and type(customCenter) is not list:
        if len(customCenter) != 2:
            raise TypeError("If custom center is specified the 'customCenter' argument must be passed as a list of two, i.e. [x,y]")
    if type(plot) is not type(True):
        raise TypeError("Plot argument should be specified as bool (True/False)")
    if len(vI)==0:
        sys.stderr.write("Passed index contains no events") 
        return []
    if center.lower() == 'density':
        center=ag.getHighestDensityPoint(fcsDF, xCol, yCol, vI)

    elif center.lower() == 'centroid':
        center=None
    else:
        center=customCenter
        
    if plot:
        fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')

    center, eigen1, eigen2 = ag.getPCs(fcsDF, xCol, yCol, center, vI)
    center, PC1, PC2 = ag.getPCSemiAxis(center, eigen1, eigen2, widthScale, heightScale)
    width=ag.getVectorLength(center, PC1)
    height=ag.getVectorLength(center, PC2)
    angle=ag.calculateAngle(center, PC1)

    result=ag.gateEllipsoid(fcsDF, xCol, yCol,xCenter=center[0],yCenter=center[1], majorAxis=[eigen1[1],eigen1[2]], minorAxis=[eigen2[1],eigen2[2]],majorRadii=width, minorRadii=height,vI=vI)

    if plot:
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

def getDensityFunc(fcsDF, xCol,vI=sentinel, sigma=3, bins=300, scale='linear', T=1000):
    data=ag.getGatedVector(fcsDF, xCol, vI, return_type="nparray")
    if scale=='logish':
        BinEdges=ag.logishBin(data,bins,T)
        histo = np.histogram(data, BinEdges)
    else:    
        histo=np.histogram(data, bins)
    smoothedHisto=ag.gaussian_filter1d(histo[0].astype(float),sigma)
    #vHisto=np.linspace(min(histo[1]),max(histo[1]),bins)
    return smoothedHisto, histo[1]

def valleySeek(fcsDF, xCol, vI=sentinel, interval=['start','end'], sigma=3, bins=300, scale='linear', T= 1000):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n")  
        return (interval[0]+interval[1])/2
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if type(interval) is not list:
        raise ValueError("Interval must be specified as list of two: [x,y].\nInterval can be half open to either side, i.e. ['start',y] or [x,'end'].")
    if len(interval)!=2:
        raise ValueError("Interval must be specified as list of two: [x,y].\nInterval can be half open to either side, i.e. ['start',y] or [x,'end'].")

    if scale=='logish':
        smoothedHisto, binData=getDensityFunc(fcsDF,xCol, vI, sigma, bins, scale='logish',T=T)
    else:
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
        sys.stderr.write("Specified interval is too narrow (Not enough data points to find a valley)\n")
        #RAISE!?
        return (interval[0]+interval[1])/2
    minVal=np.inf
    minValIndex=0
    for index in vIndicies:
        if smoothedHisto[index] < minVal:
            minVal=smoothedHisto[index]
            minValIndex=index
    return (binData[minValIndex+1]+binData[minValIndex])/2
            
def quadGate(fcsDF, xCol, yCol, xThresh, yThresh, vI=sentinel, scale='linear',T=1000):
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return [],[],[],[]
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
        errStr=str(counter)+" quadrant(s) contain no events\n"
        sys.stderr.write(errStr)
    if counter==4:
        sys.stderr.write("No quadrant contains events\n")
        return [],[],[],[]
    if plot:
        if scale=='logish':
            fig,ax=ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='auto', scale=scale, thresh=T)    
        else:
            fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')
        ag.addAxLine(fig,ax,xThresh,'vertical',scale=scale, T=T)
        ag.addAxLine(fig,ax,yThresh,'horisontal',scale=scale, T=T)
        ag.plt.show()       
    return vTopLeft, vTopRight, vBottomRight, vBottomLeft

def axisStats(fcsDF, xCol, vI=sentinel,bins=300, scale='linear',T=1000):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return 0,0,0
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if scale.lower()=='linear':
        x=ag.getGatedVector(fcsDF,xCol, vI, return_type="nparray")
    elif scale.lower()=='logish':
        x=ag.getGatedVector(fcsDF,xCol, vI, return_type="nparray")
        x=ag.logishTransform(x,T)
    
    histo=np.histogram(x, bins)
        
    mean=np.mean(x)
    sigma=np.std(x)
    maxIndex=np.argmax(histo[0])
    
    if isinstance(maxIndex, np.ndarray):
        maxVal=(histo[1][maxIndex[0]]+histo[1][maxIndex[0]+1])/2
    else:
        maxVal=(histo[1][maxIndex]+histo[1][maxIndex+1])/2
    
    if scale.lower()=='logish':
        result=ag.inverseLogishTransform([mean, sigma, maxVal],T)
        mean=result[0]
        sigma=abs(result[1])
        maxVal=result[2]
    return mean, sigma, maxVal

def gateCorner(fcsDF, xCol, yCol, xThresh, yThresh, xOrientation='upper', yOrientation='upper', vI=sentinel, bins=300, scale='linear', T=1000):
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False    
    if vI is sentinel:
        vI=fcsDF.index
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if xOrientation not in ["upper","lower"] or yOrientation not in ["upper","lower"]:
        raise TypeError("Specify desired population for xOrientation and yOrientation, 'upper' or 'lower' in regard to set thresholds")
    if len(vI)<ag.minCells:
        sys.stderr.write("Passed index contains no events\n") 
        return []  
    
    if xOrientation.lower() == "upper":
        if yOrientation.lower() == "upper":
            vOutput=fcsDF[(fcsDF[xCol]>=xThresh)&(fcsDF[yCol]>=yThresh)].index
        else:
            vOutput=fcsDF[(fcsDF[xCol]>=xThresh)&(fcsDF[yCol]<yThresh)].index
    else:
        if yOrientation.lower() == "upper":
            vOutput=fcsDF[(fcsDF[xCol]<xThresh)&(fcsDF[yCol]>=yThresh)].index
        else:
            vOutput=fcsDF[(fcsDF[xCol]<xThresh)&(fcsDF[yCol]<yThresh)].index
    
    vOutput=list(set(vOutput).intersection(vI))
    
    if plot:
        fig,ax = ag.plotHeatmap(fcsDF, xCol, yCol, vI, scale=scale,thresh=T)
        vX,vY=ag.getGatedVectors(fcsDF,xCol, yCol, vOutput, return_type="nparray")
        xmin=min(vX)
        xmax=max(vX)
        ymin=min(vY)
        ymax=max(vY)
        if xOrientation.lower() == "upper":
            if yOrientation.lower() == "upper":
                ag.addLine(fig,ax, [xmin,ymin], [xmin, ymax],scale=scale, T=T)
                ag.addLine(fig,ax, [xmin,ymin], [xmax, ymin],scale=scale, T=T)
            else:
                ag.addLine(fig,ax, [xmin,ymax], [xmax, ymax],scale=scale, T=T)
                ag.addLine(fig,ax, [xmin,ymax], [xmin, ymin],scale=scale, T=T)
        else:
            if yOrientation.lower() == "upper":
                ag.addLine(fig,ax, [xmax,ymin], [xmax, ymax],scale=scale, T=T)
                ag.addLine(fig,ax, [xmax,ymin], [xmin, ymin],scale=scale, T=T)
            else:
                ag.addLine(fig,ax, [xmax,ymax], [xmax, ymin],scale=scale, T=T)
                ag.addLine(fig,ax, [xmax,ymax], [xmin, ymax],scale=scale, T=T)
        plt.show()
        plt.clf()
        ag.plotHeatmap(fcsDF, xCol, yCol, vOutput, scale=scale)
        plt.show()
    reportGateResults(vI, vOutput)
    return vOutput


def customQuadGate(fcsDF, xCol, yCol,threshList, vI=sentinel, scale='linear',T=1000):
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return []
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if not isinstance(threshList, list):
        raise TypeError("threshList argument must be a list instance with [xbottom, xtop, yleft, yright] thresholds")
    if not len(threshList)==4:
        raise ValueError("threshList must contain 4 thresholds; [xbottom, xtop, yleft, yright]")
    if not all(isinstance(i,(float,int)) for i in threshList):
        raise TypeError("ThreshList elements must be float or int")
    if not (threshList[0]==threshList[1] or threshList[2]==threshList[3]):
        raise ValueError("Invalid values in threshList, one axis must be fix.\nEither xbottom must be equal to xtop or yleft must be equal to yright")

    vX, vY = ag.getGatedVectors(fcsDF, xCol, yCol, vI, return_type="nparray")
    assert(len(vX)==len(vY))
    vTopLeft=[]
    vTopRight=[]
    vBottomRight=[]
    vBottomLeft=[]
    if threshList[0]==threshList[1]:
        fix='x'
    else:
        fix='y'
    if fix.lower()=='x':
        xTopThresh = xBottomThresh = threshList[0]
        yRightThresh = threshList[3]
        yLeftThresh = threshList[2]
    else:
        yLeftThresh = yRightThresh = threshList[2]
        xTopThresh = threshList[1]
        xBottomThresh = threshList[0]
        
    for x,y, index in zip(vX, vY, vI):
        if y >= yLeftThresh and x < xTopThresh:
            vTopLeft.append(index)
        elif y >= yRightThresh and x >= xTopThresh:
            vTopRight.append(index)
        elif y < yLeftThresh and x < xBottomThresh:
            vBottomLeft.append(index)
        elif y < yRightThresh and x >= xBottomThresh:
            vBottomRight.append(index)
        else:
            raise RuntimeError("Unhandled index case in customQuadGate")
    assert len(vI) == (len(vBottomRight)+len(vBottomLeft)+len(vTopLeft)+len(vTopRight))
    counter=0
    for event in [len(vTopLeft),len(vTopRight),len(vBottomRight),len(vBottomLeft)]:
        if event == 0:
            counter=counter+1
    if counter != 0:
        errStr=str(counter)+" quadrant(s) contain no events\n"
        sys.stderr.write(errStr)
    if counter==4:
        sys.stderr.write("No quadrant contains events\n")
        return None
    if plot:
        if scale=='logish':
            fig,ax=ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='auto', scale=scale, thresh=T)    
        else:
            fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if fix.lower()=='x':
            ag.addLine(fig,ax,[xTopThresh,ylim[1]],[xTopThresh,ylim[0]],scale=scale, T=T)
            ag.addLine(fig,ax,[xlim[0],yLeftThresh],[xTopThresh,yLeftThresh],scale=scale, T=T)
            ag.addLine(fig,ax,[xTopThresh,yRightThresh],[xlim[1],yRightThresh],scale=scale, T=T)
        else:  
            ag.addLine(fig,ax,[xlim[0],yRightThresh],[xlim[1],yRightThresh],scale=scale, T=T)
            ag.addLine(fig,ax,[xTopThresh,ylim[1]],[xTopThresh,yLeftThresh],scale=scale, T=T)
            ag.addLine(fig,ax,[xBottomThresh,ylim[0]],[xBottomThresh,yLeftThresh],scale=scale, T=T)
        ag.plt.show()       
    return vTopLeft, vTopRight, vBottomRight, vBottomLeft

#TODO
def backGate(fcsDF, xCol, yCol, vI=sentinel, backPop=sentinel, markersize=2, scale='linear',xscale='linear',yscale='linear',T=1000):
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if vI is sentinel:
        sys.stderr.write("No vI passed, please pass a population to backgate")
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return None
    if backPop is sentinel:
        sys.stderr.write("No backPop passed setting backPop to full index") 
        backPop=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events") 
        return None
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    
    if scale=='logish':
        fig,ax=ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='auto', scale=scale, thresh=T)    
    else:
        fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')
    x,y=ag.getGatedVectors(fcsDF,xCol,yCol,backPop, return_type='nparray')
    if scale=='logish':
        xscale='logish'
        yscale='logish'
    if xscale=='logish':
        xview=ax.get_xlim()
        vmin=xview[0]
        vmax=xview[1]
        x=ag.convertToLogishPlotCoordinates(x,vmin,vmax,T)
    if yscale=='logish':
        yview=ax.get_ylim()
        vmin=yview[0]
        vmax=yview[1]
        y=ag.convertToLogishPlotCoordinates(y,vmin,vmax,T)
    #color is 'pinkish red'
    ax.plot(x,y,'o',color='#f10c45',markersize=markersize)
    
    return None
