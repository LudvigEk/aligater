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
import math
import sys
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from scipy.ndimage.filters import gaussian_filter1d

#AliGater imports
import aligater.AGConfig as agconf
from aligater.AGPlotRoutines import plotHeatmap, plot_gmm, addLine, addAxLine, transformWrapper, convertToLogishPlotCoordinates, convertToBiLogPlotCoordinates, logishBin, logishTransform, bilogBin, bilogTransform, inverseLogishTransform, inverseBilogTransform, inverseTransformWrapper
from aligater.AGCython import gateEllipsoid
from aligater.AGClasses import AGgate, AGsample
from aligater.AGFileSystem import getGatedVector, getGatedVectors, reportGateResults, invalidAGgateParentError, invalidSampleError, filePlotError, AliGaterError, markerError

def heatmap(fcs, xmarker, ymarker, population, *args, **kwargs):
    #User friendly wrapper for plotHeatmap
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in heatmap:")
    if not isinstance(population,AGgate):
        raise AliGaterError('in heatmap:','population had an unexpected type, expected AGClasses.AGgate, found '+str(type(population)))
    else:
        vI=population()
    fcsDF=fcs()
    if len(vI)==0:
        sys.stderr.write("Passed index contains no events\n")  
        return
    if 'bins' in kwargs:
            if not isinstance(kwargs['bins'],int):
                raise AliGaterError("in heatmap: ","bins parameter must be int, found: "+str(type(kwargs['bins'])))
            else:
                bins=kwargs['bins']
    else:
        bins=300
        
    if 'scale' in kwargs:
            if not isinstance(kwargs['scale'],str):
                raise AliGaterError("in heatmap: ","scale parameter must be str, found: "+str(type(kwargs['scale'])))
            else:
                scale=kwargs['scale']
    else:
        scale='linear'
    
    if 'xscale' in kwargs:
            if not isinstance(kwargs['xscale'],str):
                raise AliGaterError("in heatmap: ","xscale parameter must be str, found: "+str(type(kwargs['xscale'])))
            else:
                xscale=kwargs['xscale']
    else:
        xscale='linear'
    if 'yscale' in kwargs:
            if not isinstance(kwargs['yscale'],str):
                raise AliGaterError("in heatmap: ","yscale parameter must be str, found: "+str(type(kwargs['yscale'])))
            else:
                yscale=kwargs['yscale']
    else:
        yscale='linear'
        
    if 'aspect' in kwargs:
            if not isinstance(kwargs['aspect'],str):
                raise AliGaterError("in heatmap: ","aspect parameter must be str, found: "+str(type(kwargs['aspect'])))
            else:
                aspect=kwargs['aspect']
    else:
        aspect='auto'
        
    if 'thresh' in kwargs:
            if not isinstance(kwargs['thresh'],(int,float)):
                raise AliGaterError("in heatmap: ","thresh parameter must be float or int, found: "+str(type(kwargs['thresh'])))
            else:
                thresh=kwargs['thresh']
    else:
        thresh=1000
    
    return plotHeatmap(fcsDF, xmarker, ymarker, vI, bins, scale, xscale, yscale, thresh, aspect)

def gmm2D(fcs, xCol, yCol, nOfComponents, parentGate=None, scale='linear', T=1000, *args, **kwargs):
    """
    Fits a scikit.learn GaussianMixture object to the data and returns the gmm object.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol, yCol : str
        Marker labels.
    nOfComponents : int
        Number of components to use in the mixture model.
    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    args, kwargs : 
        optional arguments passed on to scipy.ndimage.filters.GaussianMixture, see it's sklearn documentation for options.

    **Returns**

    GaussianMixture object fit to the data.

    **Examples**

    None currently.
    """
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError("in gmm2D: ")
    else:
        vI=parentGate()
    if len(vI)<5:
        sys.stderr.write("WARNING, in gmm2D: Too few events, cannot fit mixture model, returning unfitted GMM object.\n")         
        return GaussianMixture(n_components=nOfComponents,*args,**kwargs) 
    
    if not isinstance(fcs, AGsample):
        raise invalidSampleError("in gmm2D: ")
    else:
        fcsDF=fcs()
    
    vX = getGatedVector(fcsDF,xCol,vI=vI,return_type="nparray")
    vY = getGatedVector(fcsDF,yCol,vI=vI,return_type="nparray")
    if scale.lower() != 'linear':
        vX=transformWrapper(vX, T, scale)
        vY=transformWrapper(vY, T, scale)
    fcsArray=np.array([vX,vY]).T
    gmm = GaussianMixture(n_components=nOfComponents,*args,**kwargs)
    gmm.fit(fcsArray)
    return gmm

def gateGMM(fcs, name, xCol, yCol, gmm, parentGate=None, sigma=1, widthScale=1, heightScale=1, update=False, QC=False, scale='linear', T=1000):
    """
    Function that can interpret and gate data based on a GaussianMixture object from sklearn.mixture
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
        
    xCol, yCol : str
        Marker labels.
        
    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
        
    sigma : float, optional, default: 1
        Number of standard deviations to scale the mixture model with.
        
    update : bool, optional, default: False
        If True will add the resulting gated population(s) to the sample objects gate list in adition to returning the gate object.\n
        If False (default), returns an AGgate object without adding it to the sample object.
        
    QC : bool, optional, default: False
        If True, adds a downsampled image of the gating view to the gate object. These can be collected by an AGExperiment object if it's QC flag is also True.
        
    **Returns**

    AGClasses.AGgate object

    **Examples**

    None currently.
    """
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError("in gateGMM: ")
    else:
        vI=parentGate()
    if not isinstance(gmm, GaussianMixture):
        raise TypeError("gmm argument must be a sklearn.mixture.GaussianMixture object")
    if not isinstance(fcs, AGsample):
        raise invalidSampleError("in gateGMM: ")
    else:
        fcsDF=fcs()    
    if len(vI)<5:
        sys.stderr.write("WARNING, in gateGMM: Passed parent population to "+name+" contains too few events, returning empty gate.\n") 
        outputGate=AGgate([],parentGate,xCol,yCol,name)
        if update:
            fcs.update(outputGate, QC=QC)         
        return outputGate
    
    if plot:
        fig,ax = plotHeatmap(fcsDF, xCol, yCol, vI, scale=scale, thresh=T, aspect='auto')
    else:
        ax=None
    vEllipses = plot_gmm(fcsDF,xCol, yCol, vI, gmm, sigma, ax)
    
    
    vOutput=[]
    #Gate all overlapping ellipses individually
    for ellipses in vEllipses:
        xCenter=ellipses[0][0]
        yCenter=ellipses[0][1]
        width=ellipses[1]
        height=ellipses[2]
        angle=np.radians(ellipses[3])
        majorAxis=[width*np.cos(angle), width*np.sin(angle)] / np.linalg.norm([width*np.cos(angle), width*np.sin(angle)])
        minorAxis = [height*np.cos(angle+np.pi/2), height*np.sin(angle+np.pi/2)] / np.linalg.norm([height*np.cos(angle+np.pi/2), height*np.sin(angle+np.pi/2)])
        vTmp = gateEllipsoid(fcsDF=fcsDF, xCol=xCol, yCol=yCol, xCenter=xCenter, yCenter=yCenter,majorAxis=majorAxis.tolist(), minorAxis=minorAxis.tolist(), majorRadii= widthScale*(width/2), minorRadii=heightScale*(height/2), vI=vI, info=False, scale=scale, T=T)
        vOutput.extend(vTmp)
    #The result is all unique events that was inside any ellipse
    vResult=list(set(vOutput))

    reportGateResults(vI, vResult)
    outputGate=AGgate(vResult, parentGate, xCol, yCol, name)
    if plot:
        fig, ax = plotHeatmap(fcsDF, xCol, yCol, vResult, scale=scale, thresh=T)
    return outputGate
    
    

def getPCs(fcsDF, xCol, yCol, centerCoord=None, vI=None, scale='linear', T=1000):
    #****************INTERNAL*****************
    if not xCol in fcsDF.columns:
        raise NameError("xCol not in passed dataframe's columns")
    if not yCol in fcsDF.columns:
        raise NameError("yCol not in passed dataframe's columns")
    if xCol==yCol:
        raise NameError("xCol and yCol cannot be the same")
    if vI is None:
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
    
    if scale.lower() != 'linear':
        vX=transformWrapper(vX, T, scale)
        vY=transformWrapper(vY, T, scale)
    
    if not len(vX)==len(vY):
        raise AliGaterError("Unequal amount of data points for "+str(xCol)+" and "+str(yCol),"in getPCs(internal): ")
    
    if bManualCenter:
        meanX=centerCoord[0]
        meanY=centerCoord[1]
    else:
        meanX=np.mean(vX)
        meanY=np.mean(vY)
    
    #Center dataset around the centroid, or use a custom center    
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
    
    #eigenvalues
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
    #****************INTERNAL*****************
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
    #****************INTERNAL*****************
    if not all(isinstance(i, list) for i in [lStartCoordinate, lEndCoordinate]):
        raise TypeError("Input arguments for getVectorLength (lStartCoordinate, lEndCoordinate) must be list.")
    if not len(lStartCoordinate)==len(lEndCoordinate)==2:
        raise ValueError("Input arguments for getVectorLength (lStartCoordinate, lEndCoordinate) must be lists containing two elements each.")
    length=np.sqrt(np.sum(np.square(np.subtract(lEndCoordinate,lStartCoordinate))))
    return length

def calculateAngle(lStartCoordinate, lEndCoordinate):
    #****************INTERNAL*****************
    #Angle in radians
    if not all(isinstance(i, list) for i in [lStartCoordinate, lEndCoordinate]):
        raise TypeError("Input arguments for getVectorLength (lStartCoordinate, lEndCoordinate) must be list.")
    if not len(lStartCoordinate)==len(lEndCoordinate)==2:
        raise ValueError("Input arguments for getVectorLength (lStartCoordinate, lEndCoordinate) must be lists containing two elements each.")
    angle=math.atan(np.subtract(lEndCoordinate,lStartCoordinate)[1]/np.subtract(lEndCoordinate,lStartCoordinate)[0])
    
    return angle

def calculateNormVector(lStartCoordinate, angle):
    #****************INTERNAL*****************
    #Angle in radians
    x0=lStartCoordinate[0]
    y0=lStartCoordinate[1]
    x1=math.cos(angle)+x0
    y1=math.sin(angle)+y0
    lEndCoordinate=[x1,y1]
    return lEndCoordinate

def getHighestDensityPoint(fcs, xCol, yCol, parentGate=None, bins=300, scale='linear', T=1000):
    """
    Returns coordinates for the point in the view with the highest number of events.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in a sample object.
    xCol, yCol : str
        Marker labels.
    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    bins : int, optional, default: 300
        Resolution of the heatmap used to calculate the highest density point
        
    **Returns**

    List-like
        Returns approximate coordinates of the highest density point; [x-coord, y-coord].\n
        Precision can be increased by increasing the resolution (the bins parameter)
    
    **Examples**

    None currently.
    """
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in getHighestDensityPoint:")
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError('in getHighestDensityPoint:')
    else:
        vI=parentGate()
    fcsDF=fcs()
    if (xCol not in fcsDF.columns or yCol not in fcsDF.columns):
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    vX=getGatedVector(fcsDF, xCol, vI, return_type="nparray")
    vY=getGatedVector(fcsDF, yCol, vI, return_type="nparray")
    
    if scale.lower() != 'linear':
        vX=transformWrapper(vX, T, scale)
        vY=transformWrapper(vY, T, scale)
    
    heatmap, xedges, yedges = np.histogram2d(vX, vY, bins)
    xmax=np.amax(vX)
    xmin=np.amin(vX)
    ymax=np.amax(vY)
    ymin=np.amin(vY)
    highestPoint=np.unravel_index(heatmap.argmax(), heatmap.shape)
    #Re-estimate original index, note +- range/bins error
    xCoord=highestPoint[0]*(xmax-xmin)/bins + xmin
    yCoord=highestPoint[1]*(ymax-ymin)/bins + ymin
    
    outputCoords = [xCoord, yCoord]
    if scale.lower() != 'linear':
        outputCoords = inverseTransformWrapper(outputCoords, T, scale)
        
    return outputCoords
    


def gatePC(fcs, xCol, yCol, name, parentGate=None, widthScale=1, heightScale=1, center='centroid', customCenter=None, filePlot=None, scale='linear', T=1000, **kwargs):
    """
    Function that performs a 2D principal component analysis and gates an ellipse based on the results.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol, yCol : str
        Marker labels.
    name : str
        Name to the resulting gated population.F
    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    widthScale, heightScale : float, optional, default: 1
        Number of standard deviations to scale eigenvectors with, corresponding to the width and height of the ellipse.
    center : str, optional, default: 'centroid'
        Where to center the image for PC analysis, options are 'center', 'centroid' or 'custom'\n
        Data will temporarily be centered around this point for the PC analysis.
    customCenter : None or list-like, optional, default: None
        Only used if center option is 'custom'. Then takes a list-like of two being the coordinates of the center point.
    filePlot : str, optional, default: None
        Option to plot the gate to file to specified path. \n
        Warning: might overwrite stuff.

    **Returns**
    
    AGClasses.AGgate object
        
    **Examples**

    None currently.
    """
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")
    if parentGate is None:
        vI=fcs.full_index()
    elif not parentGate.__class__.__name__ == "AGgate":
        raise TypeError("invalid AGgate object")
    else:
        vI=parentGate()
    fcsDF=fcs()
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
    if len(vI)<5:
        sys.stderr.write("WARNING, in gatePC: Passed parent population to "+name+" contains too few events, returning empty gate.\n") 
        return AGgate([], parentGate, xCol, yCol, name)
    if center.lower() == 'density':
        center=getHighestDensityPoint(fcs, xCol, yCol, parentGate, scale=scale, T=T)
    elif center.lower() == 'centroid':
        center=None
    else:
        if scale.lower() != 'linear':
            center=customCenter
        else:
            center=transformWrapper(customCenter, T, scale)
        
    if plot or filePlot is not None:
        fig, ax = plotHeatmap(fcsDF, xCol, yCol, vI, scale=scale, thresh=T)

    center, eigen1, eigen2 = getPCs(fcsDF, xCol, yCol, center, vI, scale=scale, T=T)

    center, PC1, PC2 = getPCSemiAxis(center, eigen1, eigen2, widthScale, heightScale)
    width=getVectorLength(center, PC1)
    height=getVectorLength(center, PC2)
    angle=calculateAngle(center, PC1)
    if 'adjustAngle' in kwargs:
        #Collect requested adjustment
        adjustAngle=kwargs['adjustAngle']
        assert isinstance(adjustAngle,(float, int))
        #Recalculate eigen 1
        adjustAngle=math.radians(adjustAngle)
        angle=angle+adjustAngle
        new_eigen1=calculateNormVector([0,0], adjustAngle)
        #Recalculate eigen 2
        secondAngle=calculateAngle(center, PC2)
        secondAngle=secondAngle+adjustAngle
        new_eigen2=calculateNormVector([0,0],secondAngle)

        eigen1[1]=new_eigen1[0]
        eigen1[2]=new_eigen1[1]
        eigen2[1]=new_eigen2[0]
        eigen2[2]=new_eigen2[1]
       
    
    result=gateEllipsoid(fcsDF, xCol, yCol,xCenter=center[0],yCenter=center[1], majorAxis=[eigen1[1],eigen1[2]],majorRadii=width ,minorAxis=[eigen2[1],eigen2[2]], minorRadii=height,vI=vI, scale=scale, T=T)

    if plot or filePlot is not None:
        #addLine(fig, ax, center, PC1)
        #addLine(fig, ax, center, PC2)
        ax.add_patch(Ellipse(center, 2*width, 2*height, np.degrees(angle), fill=False, edgecolor='#FF0000', linestyle='dashed'))
        if filePlot is not None:
            plt.savefig(filePlot)
            if not plot:
                plt.close(fig)
        if plot:
            plt.show()
            plotHeatmap(fcsDF, xCol, yCol, result, scale=scale, thresh=T)
            plt.show()
            plt.clf()
    if parentGate is not None:
        outputGate=AGgate(result, parentGate, xCol, yCol, name)
    else:
        outputGate=AGgate(result, None, xCol, yCol, name)
    return outputGate

def getVectorCoordinates(length, angle):
    #*********Internal****************
    theta=math.degrees(angle)
    y = length*math.asin(theta)
    x = length*math.acos(theta)
    return[x,y]

def getDensityFunc(fcsDF, xCol,vI=None, sigma=3, bins=300, scale='linear', T=1000):
    #*********Internal****************
    if vI is None:
        vI=fcsDF.full_index()
    data=getGatedVector(fcsDF, xCol, vI, return_type="nparray")
    if scale.lower()=='logish':
        BinEdges=logishBin(data,bins,T)
        histo = np.histogram(data, BinEdges)
    elif scale.lower()=='bilog':
        BinEdges=bilogBin(data, bins, T)
        histo = np.histogram(data, BinEdges)
    else:    
        histo=np.histogram(data, bins)
    smoothedHisto=gaussian_filter1d(histo[0].astype(float),sigma)

    return smoothedHisto, histo[1]

def valleySeek(fcs, xCol, parentGate=None, interval=['start','end'], sigma=3, bins=300, require_local_min=False, scale='linear', T= 1000):
    """
    Function that finds the least dense point in a given interval by searching a smoothed density function.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol : str
        Marker label.
    name : str
        Name to the resulting gated population.
    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    interval : list-like, optional, default: ['start','end']
        Interval to limit the search, defaults to entire axis.\n
        Some examples: [5, 'end'], ['start', 6800], [30, 1500]
        Accepts text-strings 'start' and 'first' as synonyms and similarly for 'end', 'last'
    sigma : float, optional, default: 3
        Smoothing factor of density function (kernel smooth).
    bins : int, optional, default: 300
        Number of bins in density histogram.
    scale : str, optional, default: 'linear'
        If plotting enabled, which scale to be used on axis.
    T : int, optional, default: 1000
        If plotting enabled and scale is logish, the threshold for linear-loglike transition

    **Returns**

    float
        Coordinate on axis with lowest density in given interval.
        
    
    .. note::
            If less than 5 events are passed in parentGate, returns mid interval without attempting to valleyseek\n
            Since the lowest density point is estimated on a smoothed histogram/density func there is an built-in error of +- ( bin width / 2 )
    
    **Examples**

    None currently.
    """
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in valleySeek:")
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError('in valleySeek:')
    else:
        vI=parentGate()
    fcsDF=fcs()
    if len(vI)<5:
        sys.stderr.write("WARNING, in valleySeek: Passed index contains too few events, defaulting to mid-interval\n")  
        return (interval[0]+interval[1])/2
    if xCol not in fcsDF.columns:
        raise AliGaterError("in valleySeek: ","Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if type(interval) is not list:
        raise AliGaterError("in valleySeek: ","Interval must be specified as list of two: [x,y].\nInterval can be half open to either side, i.e. ['start',y] or [x,'end'].")
    if len(interval)!=2:
        raise AliGaterError("in valleySeek: ","Interval must be specified as list of two: [x,y].\nInterval can be half open to either side, i.e. ['start',y] or [x,'end'].")
    if not any(isinstance(i,(float,int, str)) for i in interval):
        raise(AliGaterError("in valleySeek: ","Interval element had an unexpected type"))


    vX = getGatedVector(fcsDF, gate=xCol, vI=vI,return_type="nparray")
    if scale.lower()!='linear':
        vX = transformWrapper(vX, scale=scale, T=T)
    histo, binData = np.histogram(vX,bins)
    smoothedHisto=gaussian_filter1d(histo.astype(float),sigma)
    
    if type(interval[0]) is str:
        if interval[0].lower() in ['start', 'first']:
            interval[0]=min(binData) 
        else:
            raise AliGaterError("in valleySeek: ","limit specified as string but option unrecognized, expected 'first' or 'start', found "+interval[0].lower())
    else:
        if scale.lower!='linear':
            interval[0]=transformWrapper([interval[0]],scale=scale, T=T)[0]
    if type(interval[1]) is str:
        if interval[1].lower() in ['end', 'last']:
            interval[1]=max(binData) 
        else:
            raise AliGaterError("in valleySeek: ","limit specified as string but option unrecognized, expected 'last' or 'end', found "+interval[1].lower())
    else:
        if scale.lower!='linear':
            interval[1]=transformWrapper([interval[1]],scale=scale, T=T)[0]
    
    if interval[1] > max(binData):
        interval[1] = max(binData)

    vIndicies=[]
    #binData is the bin edges
    lowerLimit = interval[0]
    upperLimit = interval[1]
    for index, x in np.ndenumerate(binData):
        #Note the non-inclusive upper bound, critical to stay in-bound of array index
        if x >= lowerLimit and x < upperLimit:
            vIndicies.append(index[0])

    if len(vIndicies)<=3:
        sys.stderr.write("WARNING, in valleySeek: Specified interval is too narrow, defaulting to mid-interval\n")
        result = (interval[0]+interval[1])/2
        if scale.lower()!='linear':
            result = inverseTransformWrapper([result],scale=scale, T=T)[0]
        return result
    
    if not require_local_min:
        minVal=np.inf
        minValIndex=0
        for index in vIndicies:
            if smoothedHisto[index] < minVal:
                minVal=smoothedHisto[index]
                minValIndex=index
                
    if require_local_min: 
        minVal=np.inf
        minValIndex=0
        for i in np.arange(1,len(vIndicies)-1,1):
            value=smoothedHisto[vIndicies[i]]
            if value < minVal and value < smoothedHisto[vIndicies[i-1]] and value < smoothedHisto[vIndicies[i+1]]:
                minVal=value
                minValIndex=vIndicies[i]         
        if minVal==np.inf:
            #TODO, maybe raise this, or warning. If warning, reasonable return? 
            #Let user set return behaviour (raise - min - max - mid or val)?
            sys.stderr.write("in valleySeek: Local min requested, but none found. Returning infinity.")
            return np.inf
    
    result=(binData[minValIndex+1]+binData[minValIndex])/2
    if scale.lower()!='linear':
            result = inverseTransformWrapper([result],scale=scale, T=T)[0]
    return result
            
def quadGate(fcs, names, xCol, yCol, xThresh, yThresh, parentGate=None, scale='linear',T=1000, filePlot=None, QC=False):
    """
    Function that gates four populations from one view by drawing a cross.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in a sample object.
    
    names : list-like
        | list of string with four names for the output gated populations in clockwise order from top-left;
        | Top-left, top-right, bottom-right, bottom-left
        
    xCol, yCol : str
        Marker labels.
        
    xThresh : float
        Threshold for vertical line.

    yThresh : float
        Threshold for horisontal line.

    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
        
    scale : str, optional, default: 'linear'
        If plotting enabled, which scale to be used on both axis.
        
    T : int, optional, default: 1000
        If plotting enabled and scale is logish, the threshold for linear-loglike transition

    filePlot : str, optional, default: None
        Option to plot the gate to file to specified path.\n
        Warning: might overwrite stuff.    
        
    QC : bool, optional, default: False
        If True, adds a downsampled image of the gating view to the gate object. These can be collected by an AGExperiment object if it's QC flag is also True.
    
    **Returns**

    AGClasses.AGgate, AGClasses.AGgate, AGClasses.AGgate, AGClasses.AGgate
        Returns AGClasses.AGgate objects for the four gated populations.\n 
        In clockwise order; top-left, top-right, bottom-right, bottom-left    
    
    .. note::
            All limits are considered greater than inclusive (<=) and less than exclusive (>)\n
            If QC is requested, the downsampled image will be same of all the gated populations. Only one image will be saved, with its name given by the top-left population.
        
    **Examples**

    None currently.
    """
    
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in quadGate:")
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError('in quadGate:')
    else:
        vI=parentGate()
    fcsDF=fcs()
    if len(vI)<5:
        sys.stderr.write("WARNING: in quadGate: Passed population ("+str(parentGate.name)+") contains <5 events, returning empty gates\n") 
        return AGgate([],parentGate,xCol,yCol,names[0]), AGgate([],parentGate,xCol,yCol,names[1]), AGgate([],parentGate,xCol,yCol,names[2]), AGgate([],parentGate,xCol,yCol,names[3])
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if not all(isinstance(i,(float, int)) for i in [xThresh, yThresh]):
        raise TypeError("xThresh, yThresh must be specified as integer or floating-point values")
    if isinstance(names, list):
        if not len(names)==4:
            raise AliGaterError("The names parameter doesn't have exactly four elements","in quadGate:")
    else:
        raise AliGaterError("Unexpected type of names parameter, expected "+str(type(list))+" found "+str(type(names)),"in quadGate: ")
    if not all(isinstance(i,(str)) for i in names):
        raise AliGaterError("Non-str element encountered in the names list","in quadGate: ")
    vX, vY = getGatedVectors(fcsDF, xCol, yCol, vI, return_type="nparray")
    assert(len(vX)==len(vY))
    vTopLeft=[]
    vTopRight=[]
    vBottomRight=[]
    vBottomLeft=[]
    for x,y, index in zip(vX, vY, vI):
        if x <= xThresh and y > yThresh:
            vTopLeft.append(index)
        elif x > xThresh and y > yThresh:
            vTopRight.append(index)
        elif x > xThresh and y <= yThresh:
            vBottomRight.append(index)
        elif x <= xThresh and y <= yThresh:
            vBottomLeft.append(index)
        else:
            raise RuntimeError("Unhandled event case in quadGate")
    counter=0
    for event in [len(vTopLeft),len(vTopRight),len(vBottomRight),len(vBottomLeft)]:
        if event == 0:
            counter=counter+1
    if counter != 0 and counter != 4:
        errStr="WARNING: in quadGate, with parent population "+str(parentGate.name)+": "+str(counter)+" quadrant(s) contain no events\n"
        sys.stderr.write(errStr)
    if counter==4:
        sys.stderr.write("WARNING: in quadGate, with parent population "+str(parentGate.name)+":  No quadrant contains events\n")
        return AGgate([],parentGate,xCol,yCol,names[0]), AGgate([],parentGate,xCol,yCol,names[1]), AGgate([],parentGate,xCol,yCol,names[2]), AGgate([],parentGate,xCol,yCol,names[3])
    
    if plot or filePlot is not None:
        if scale=='logish':
            fig,ax=plotHeatmap(fcsDF, xCol, yCol,vI,aspect='auto', scale=scale, thresh=T)    
        else:
            fig, ax = plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')
        addAxLine(fig,ax,xThresh,'vertical',scale=scale, T=T)
        addAxLine(fig,ax,yThresh,'horisontal',scale=scale, T=T)

        if plot:
            plt.show()    
        if filePlot is not None:
            plt.savefig(filePlot)
            if not plot:
                plt.close(fig)
        plt.close(fig)  
    
    TopLeft=AGgate(vTopLeft, parentGate, xCol, yCol, names[0])
    TopRight=AGgate(vTopRight, parentGate, xCol, yCol, names[1])
    BottomRight=AGgate(vBottomRight, parentGate, xCol, yCol, names[2])
    BottomLeft=AGgate(vBottomLeft, parentGate, xCol, yCol, names[3])
    
    if agconf.ag_verbose:
        reportStr="quadGate results in clockwise order from top-left: "+str(len(vTopLeft))+", "+str(len(vTopRight))+", "+str(len(vBottomRight))+", "+str(len(vBottomLeft))+"\n"
        sys.stderr.write(reportStr)
        
    return TopLeft, TopRight, BottomRight, BottomLeft




def axisStats(fcsDF, xCol, vI=None,bins=300, scale='linear',T=1000):
    """
    Report mean, standard deviation and maximum value on axis.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol : str
        Marker label.
    vI : list-like or AGgate object
        Parent population to apply the gating to.
    bins : int, optional, default: 300
        Number of bins in density histogram.
    scale : str, optional, default: 'linear'
        The returned values can be transformed to a plotting scale; \n
        options: 'linear', 'logish'
    T : int, optional, default: 1000
        If plotting enabled and scale is logish, the threshold for linear-loglike transition

    **Returns**

    float, float, float
        mean, standard deviation, maximum value

    **Examples**

    None currently.
    """
    if vI is None:
        vI=fcsDF.index
    elif len(vI)<5:
        sys.stderr.write("WARNING, in AxisStats: Passed parent population contains too few events, returning Zero (0).\n") 
        return 0,0,0,0
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if scale.lower()=='linear':
        x=getGatedVector(fcsDF,xCol, vI, return_type="nparray")
    else:
        x=getGatedVector(fcsDF,xCol, vI, return_type="nparray")
        x=transformWrapper(x,scale=scale, T=T)

    histo=np.histogram(x, bins)

    mean=np.mean(x)
    sigma=np.std(x)
    median = np.median(x)
    maxIndex=np.argmax(histo[0])

    if isinstance(maxIndex, np.ndarray):
        maxVal=(histo[1][maxIndex[0]]+histo[1][maxIndex[0]+1])/2
    else:
        maxVal=(histo[1][maxIndex]+histo[1][maxIndex+1])/2

    #if scale.lower()=='logish':
    #    result=inverseLogishTransform([mean, sigma, maxVal],T)
    #    mean=result[0]
    #    sigma=abs(result[1])
    #    maxVal=result[2]

    return mean, median, sigma, maxVal

def EllipseGate(fcs, name, xCol, yCol, center, width, parentGate=None, height=None, angle=0, scale='linear', T=1000, filePlot=None):
    #TODO: NOT WORKING WITH SCALES!
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False    
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in EllipseGate:")
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError('in EllipseGate:')
    else:
        vI=parentGate()
    fcsDF=fcs()
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if not isinstance(center,(list, np.ndarray)):
        raise AliGaterError("in EllipseGate","center must be python list or numpy array")
    elif not len(center) == 2:
        raise AliGaterError("in EllipseGate","invalid center dim (must be length of two list-like)")       
    if height is None:
        height=width
    if width <= 0 or height <=0:
        raise AliGaterError("in EllipseGate","height/width cant be negative")       
        
    #fcsDF, str xCol, str yCol, float xCenter, float yCenter, list majorAxis, float majorRadii, list minorAxis, float minorRadii    
    
    if angle == 0: #To avoid weird rounding when angle is zero
        majorAxis = [1,0]
        minorAxis = [0,1]
    else:
        majorAxis = [width*np.cos(angle), width*np.sin(angle)] / np.linalg.norm([width*np.cos(angle), width*np.sin(angle)])
        minorAxis = [height*np.cos(angle+np.pi/2), height*np.sin(angle+np.pi/2)] / np.linalg.norm([height*np.cos(angle+np.pi/2), height*np.sin(angle+np.pi/2)])

    if scale.lower() != 'linear':
        if center[0] != 0:
            width = transformWrapper(width+center[0], scale=scale, T=T) - transformWrapper(width, scale=scale, T=T)
        else:
            width = transformWrapper(width, scale=scale, T=T)
        if center[1] != 0:
            height = transformWrapper(height+center[1], scale=scale, T=T) - transformWrapper(height, scale=scale, T=T)
        else:
            height = transformWrapper(height, scale=scale, T=T)
        center = transformWrapper(center, scale=scale, T=T)
        #majorAxis = transformWrapper(majorAxis, scale=scale, T=T)
        #minorAxis = transformWrapper(minorAxis, scale=scale, T=T)

    majorAxis=list(majorAxis)
    minorAxis=list(minorAxis)

    #if using a non-linear scale, the input center, axis and radii must be in the correct transformed coordinates.
    
    vOut = gateEllipsoid(fcsDF=fcs(), xCol=xCol, yCol=yCol, xCenter=center[0], yCenter=center[1], majorAxis = majorAxis, majorRadii = width, minorAxis = minorAxis, minorRadii=height, vI=vI, population="inner", info=False, scale=scale, T=T)
    output_gate = AGgate(vOut, parentGate, xCol, yCol, name)
    return output_gate

def gateCorner(fcs, name, xCol, yCol, xThresh, yThresh, xOrientation='upper', yOrientation='upper', Outer=False, parentGate=None, bins=300, scale='linear', T=1000, update=False, filePlot=None, QC=False):
    """
    Gates a corner in the view, with xOrientation and yOrientation parameters deciding the shape (which corner to gate).
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol, yCol : str
        Marker labels.
    name : str
        Name to the resulting gated population.
    parentGate : AGClasses.AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    xThresh, yThresh : float
        The X- and Y-axis thresholds for the gate.
    xOrientation, yOrientation : str, optional, default: 'upper'
        Defines which population is gated as positive in relation to the x- and y- thresholds. The default parameter means top right corner.
    outer : bool, optional, default: False
        If True, instead returns all events outside of the defined corner.
    bins : int, optional, default: 300
        If plotting, defines the resolution of the heatmap.
    scale : str, optional, default: 'linear'
        If plotting enabled, which scale to be used on both axis.
    T : int, optional, default: 1000
        If plotting enabled and scale is logish, the threshold for linear-loglike transition.    
    filePlot : str, optional, default: None
        Option to plot the gate to file to specified path.\n
        Warning: might overwrite stuff.
    update : bool, optional, default: False
        If True will add the resulting gated population(s) to the sample objects gate list in adition to returning the gate object.\n
        If False (default), returns an AGgate object without adding it to the sample object.
    QC : bool, optional, default: False
        If True, adds a downsampled image of the gating view to the gate object. These can be collected by an AGExperiment object if it's QC flag is also True.
        
    **Returns**

    AGClasses.AGgate object

    **Examples**

    None currently.
    """    
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False    
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in gateCorner:")
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError('in gateCorner:')
    else:
        vI=parentGate()
    fcsDF=fcs()
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if xOrientation not in ["upper","lower"] or yOrientation not in ["upper","lower"]:
        raise TypeError("Specify desired population for xOrientation and yOrientation, 'upper' or 'lower' in regard to set thresholds")
    if len(vI)<5:
        sys.stderr.write("WARNING, in gateCorner: Passed parent population to "+name+" contains too few events, returning empty gate.\n") 
        return AGgate([], parentGate, xCol, yCol, name)
    
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
    if not Outer:
        vOutput=list(set(vOutput).intersection(vI))
    else:
        vOutput=list(set(vI).difference(vOutput))
    
    if len(vOutput)==0:
        return AGgate([], parentGate, xCol, yCol, name)
    
    if plot or filePlot is not None:
        fig,ax = plotHeatmap(fcsDF, xCol, yCol, vI,bins=bins, scale=scale,thresh=T)
        vX,vY=getGatedVectors(fcsDF,xCol, yCol, vOutput, return_type="nparray")
        xmin=min(vX)
        xmax=max(vX)
        ymin=min(vY)
        ymax=max(vY)
        if xOrientation.lower() == "upper":
            if yOrientation.lower() == "upper":
                addLine(fig,ax, [xThresh,yThresh], [xThresh, ymax],scale=scale, T=T)
                addLine(fig,ax, [xThresh,yThresh], [xmax, yThresh],scale=scale, T=T)
            else:
                addLine(fig,ax, [xThresh,yThresh], [xThresh, ymin],scale=scale, T=T)
                addLine(fig,ax, [xThresh,yThresh], [xmax, yThresh],scale=scale, T=T)
        else:
            if yOrientation.lower() == "upper":
                addLine(fig,ax, [xThresh,yThresh], [xmin, yThresh],scale=scale, T=T)
                addLine(fig,ax, [xThresh,yThresh], [xThresh, ymax],scale=scale, T=T)
            else:
                addLine(fig,ax, [xThresh,yThresh], [xmin, yThresh],scale=scale, T=T)
                addLine(fig,ax, [xThresh,yThresh], [xThresh, ymin],scale=scale, T=T)
        if filePlot is not None:
            plt.savefig(filePlot)
            if not plot:
                plt.close(fig)
        if plot:
            plt.show()
            plt.clf()
            plotHeatmap(fcsDF, xCol, yCol, vOutput,bins=bins, scale=scale, thresh=T)
            plt.show()

        
    if parentGate is not None:
        outputGate=AGgate(vOutput, parentGate, xCol, yCol, name)
    else:
        outputGate=AGgate(vOutput, None, xCol, yCol, name)
    if update:
        fcs.update(outputGate, QC=QC)
    reportGateResults(vI, vOutput)
    return outputGate


def customQuadGate(fcs, names, xCol, yCol,threshList, parentGate=None, scale='linear',T=1000, filePlot=None):
    """
    A quadgate function with one axis fix and the other variable. The threshList argument decides which axis is fix and which can vary.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol, yCol : str
        Marker labels.
    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    threshList : list-like of float or int
        Requires four float or int values. These are the thresholds in each direction for the gate. \n
        In order; bottom x threshold, top x threshold, left y threshold, right y threshold.
        Either the x or y thresholds must be equal (that axis will be fix).
    scale : str, optional, default: 'linear'
        If plotting enabled, which scale to be used on both axis.
    T : int, optional, default: 1000
        If plotting enabled and scale is logish, the threshold for linear-loglike transition.    
    filePlot : str, optional, default: None
        Option to plot the gate to file to specified path. \n
        Warning: might overwrite stuff.
    
    **Returns**

    AGClasses.AGgate, AGClasses.AGgate, AGClasses.AGgate, AGClasses.AGgate
        Returns AGClasses.AGgate objects for the four gated populations.\n 
        In clockwise order; top-left, top-right, bottom-right, bottom-left
    
    .. note::
            All limits are considered greater than inclusive (<=) and less than exclusive (>)\n
            If QC is requested, the downsampled image will be same of all the gated populations. Only one image will be saved, with its name given by the top-left population.


    **Examples**

    None currently.
    """     
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError("in customQuadGate: ")
    else:
        vI=parentGate()
    if not isinstance(fcs, AGsample):
        raise invalidSampleError("in customQuadGate: ")
    else:
        fcsDF=fcs()
    if len(vI)<5:
        sys.stderr.write("WARNING: in customQuadGate: Passed population ("+str(parentGate.name)+") contains <5 events, returning empty gates\n") 
        return AGgate([],parentGate,xCol,yCol,names[0]), AGgate([],parentGate,xCol,yCol,names[1]), AGgate([],parentGate,xCol,yCol,names[2]), AGgate([],parentGate,xCol,yCol,names[3])
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")
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
    if isinstance(names, list):
        if not len(names)==4:
            raise AliGaterError("The names parameter doesn't have exactly four elements","in customQuadGate:")
    else:
        raise AliGaterError("Unexpected type of names parameter, expected "+str(type(list))+" found "+str(type(names)),"in customQuadGate: ")
    if not all(isinstance(i,(str)) for i in names):
        raise AliGaterError("Non-str element encountered in the names list","in customQuadGate: ")
    
    vX, vY = getGatedVectors(fcsDF, xCol, yCol, vI, return_type="nparray")
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
            raise RuntimeError("Unhandled event case in customQuadGate")
    assert len(vI) == (len(vBottomRight)+len(vBottomLeft)+len(vTopLeft)+len(vTopRight))
    counter=0
    for event in [len(vTopLeft),len(vTopRight),len(vBottomRight),len(vBottomLeft)]:
        if event == 0:
            counter=counter+1
    if counter != 0 and counter != 4:
        errStr="WARNING: in customQuadGate, with parent population "+str(parentGate.name)+": "+str(counter)+" quadrant(s) contain no events\n"
        sys.stderr.write(errStr)
    if counter==4:
        sys.stderr.write("WARNING: in customQuadGate, with parent population "+str(parentGate.name)+":  No quadrant contains events\n")
        return None
    if plot or filePlot is not None:
        if scale=='logish':
            fig,ax=plotHeatmap(fcsDF, xCol, yCol,vI,aspect='auto', scale=scale, thresh=T)    
        else:
            fig, ax = plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if fix.lower()=='x':
            addLine(fig,ax,[xTopThresh,ylim[1]],[xTopThresh,ylim[0]],scale=scale, T=T)
            addLine(fig,ax,[xlim[0],yLeftThresh],[xTopThresh,yLeftThresh],scale=scale, T=T)
            addLine(fig,ax,[xTopThresh,yRightThresh],[xlim[1],yRightThresh],scale=scale, T=T)
        else:  
            addLine(fig,ax,[xlim[0],yRightThresh],[xlim[1],yRightThresh],scale=scale, T=T)
            addLine(fig,ax,[xTopThresh,ylim[1]],[xTopThresh,yLeftThresh],scale=scale, T=T)
            addLine(fig,ax,[xBottomThresh,ylim[0]],[xBottomThresh,yLeftThresh],scale=scale, T=T)
        if filePlot is not None:
            plt.savefig(filePlot)
            plt.close()
        if plot:
            plt.show()
    TopLeft=AGgate(vTopLeft, parentGate, xCol, yCol, names[0])
    TopRight=AGgate(vTopRight, parentGate, xCol, yCol, names[1])
    BottomRight=AGgate(vBottomRight, parentGate, xCol, yCol, names[2])
    BottomLeft=AGgate(vBottomLeft, parentGate, xCol, yCol, names[3])
    if agconf.ag_verbose:
        reportStr="customQuadGate results in clockwise order from top-left: "+str(len(vTopLeft))+", "+str(len(vTopRight))+", "+str(len(vBottomRight))+", "+str(len(vBottomLeft))+"\n"
        sys.stderr.write(reportStr)

    return TopLeft, TopRight, BottomRight, BottomLeft

def backGate(fcs, xCol, yCol, population, background_population=None, markersize=2, scale='linear',xscale='linear',yscale='linear',T=1000, filePlot=None, color='#f10c45'):
    """
    Highlights a population onto another view/population.\n
    Typically used to see where rarer populations are located on an earlier view in a gating strategy.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in a sample object.
    xCol,yCol : str
        Marker labels.
    population : AGgate object
        Population that should be highlighted.
    background_population : AGgate object
        Background population.
    markersize : float, optional, default: 2
        Size of events of the overlayed/highlighted population.
    scale : str, optional, default: 'linear'
        Which scale to be used on axis.
    xscale : str, optional, default: 'linear'
        Which scale to be used on x-axis.
    yscale : str, optional, default: 'linear'
        Which scale to be used on y-axis.        
    T : int, optional, default: 1000
        If scale is logish, the threshold for linear-loglike transition.
    filePlot : str, optional, default: None
        Option to plot the gate to file to specified path. \n
        Warning: might overwrite stuff.
    color : str, optional, default: '#f10c45'
        Color of the highlighted population\n
        default is 'pinkish red' from XKCD's color survey https://xkcd.com/color/rgb/
    
    .. note::
            If the scale parameter is changed from default (linear) this will override any settings in xscale, yscale.

    **Returns**

    None 
    
    **Examples**

    None currently.
    """
    
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in backGate:")
        
    if not isinstance(population,AGgate):
        raise AliGaterError("Invalid AGgate object passed as population","in backGate: ")
    else:
        backPop=population()
        
    if background_population is None:
        if agconf.ag_verbose:
            sys.stderr.write("WARNING, in backGate: No background_population passed, setting background population to full index\n") 
        vI=fcs.full_index()
    elif not isinstance(background_population,AGgate):
        raise AliGaterError("Invalid AGgate object passed as background_population","in backGate: ")
    else:
        vI=background_population()
    fcsDF=fcs()
    if len(backPop)==0:
        sys.stderr.write("WARNING, in backGate: Passed population ("+str(population.name)+") contains no events\n") 
        return None
    if len(vI)==0:
        sys.stderr.write("WARNING, in backGate: Passed background population ("+str(background_population.name)+") contains no events\n") 
        return None
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise markerError("in backGate: ")
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise filePlotError("in backGate: ") 
            
    #backPop = population to highlight
    #vI = background population
    if scale!='linear':
        fig,ax=plotHeatmap(fcsDF, xCol, yCol,vI,aspect='auto', scale=scale, thresh=T)    
    else:
        fig, ax = plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')
        
    x,y=getGatedVectors(fcsDF,xCol,yCol,backPop, return_type='nparray')
    if scale=='logish':
        xscale='logish'
        yscale='logish'
    if scale=='bilog':
        xscale='bilog'
        yscale='bilog'
    if xscale!='linear':
        xview=ax.get_xlim()
        vmin=xview[0]
        vmax=xview[1]
        if xscale=='logish':
            x=convertToLogishPlotCoordinates(x,vmin,vmax,T)
        if xscale=='bilog':
            x=convertToBiLogPlotCoordinates(x,vmin,vmax,T)
    if yscale!='linear':
        yview=ax.get_ylim()
        vmin=yview[0]
        vmax=yview[1]
        if yscale=='logish':
            y=convertToLogishPlotCoordinates(y,vmin,vmax,T)
        if yscale=='bilog':
            y=convertToBiLogPlotCoordinates(y,vmin,vmax,T)
    ax.plot(x,y,'o',color=color,markersize=markersize)
    if filePlot is not None:
        plt.savefig(filePlot)
        plt.close()
    else:
        plt.show()
    return None

def gateTiltedLine(fcs, xCol, yCol, theta, name, parentGate=None, startPoint=(None,None), endLimits=(None,None), population='upper', scale='linear', xscale='linear', yscale='linear', T=1000, filePlot=None):
    """
    Gates the population from a line given by an angle (-90 < theta < 90) and optionally a startpoint and/or endlimit(s).
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in a sample object.
    xCol,yCol : str
        Marker labels.
    population : AGgate object
        Population that should be highlighted.
    theta : float/int
        The angle in degrees, (-90 < theta < 90)
    name : str
        Name of the resulting gated population.
    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    startPoint : tuple(float/int), optional, default : (None, None)
        Optional start point where to start the tilted line. 
    endLimits : tuple(float/int), optional, default : (None, None)
        Optional end limits, if the tilted line passes through EITHER the x or y limit specified by endLimits it will stop and proceed acoording to endOrientation.   
    population, str, optional, default: 'upper'
        This parameter determines which population should be returned.\n
        'upper' means any events with a value above the tresholds are returned.\n
        'lower' means any events with a value below the tresholds will be returned.\n
        The default setting means the population that's considered 'positive' in flow cytometry terms is returned.
    scale : str, optional, default: 'linear'
        Which scale to be used on axis.
    xscale : str, optional, default: 'linear'
        Which scale to be used on x-axis.
    yscale : str, optional, default: 'linear'
        Which scale to be used on y-axis.        
    T : int, optional, default: 1000
        If scale is logish, the threshold for linear-loglike transition.
    filePlot : str, optional, default: None
        Option to plot the gate to file to specified path. \n
        Warning: might overwrite stuff.

    **Returns**

    None 
    
    **Examples**

    None currently.
    """
    
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError("in gateTiltedLine: ")
    else:
        vI=parentGate()
    if not isinstance(fcs, AGsample):
        raise invalidSampleError("in gateTiltedLine: ")
    else:
        fcsDF=fcs()
    if len(vI)<5:
        sys.stderr.write("WARNING: in gateTiltedLine: Passed population ("+str(parentGate.name)+") contains <5 events, returning empty gate\n") 
        return AGgate([],parentGate,xCol,yCol,name)
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels") 
    if not isinstance(theta, (float, int)):
        raise AliGaterError("in gateTiltedLine: ","theta had an invalid dtype, expected "+str(type(float))+"/"+str(type(int))+" found: "+str(type(theta)))
    if not (-90 < theta < 90) or theta == 0:
        raise AliGaterError("in gateTiltedLine: ","theta must be between -90 and 90 degrees, non inclusive. (-90 < theta < 90) and cannot be zero (0)")
        
    if not isinstance(startPoint, tuple):
        raise AliGaterError("in gateTiltedLine: ","startPoint must be tuple containing any combination of float, int or None")
    if not isinstance(startPoint[0],(float,int)) and startPoint[0] is not None:
        raise AliGaterError("in gateTiltedLine: ","startPoint must be tuple containing any combination of float, int or None")
    if not isinstance(startPoint[1],(float,int)) and startPoint[1] is not None:
        raise AliGaterError("in gateTiltedLine: ","startPoint must be tuple containing any combination of float, int or None")        

    if not isinstance(endLimits, tuple):
        raise AliGaterError("in gateTiltedLine: ","endLimits must be tuple of float or int")
    if not isinstance(endLimits[0],(float,int)) and endLimits[0] is not None:
        raise AliGaterError("in gateTiltedLine: ","endLimits must be tuple containing any combination of float, int or None")
    if not isinstance(endLimits[1],(float,int)) and endLimits[1] is not None:
        raise AliGaterError("in gateTiltedLine: ","endLimits must be tuple containing any combination of float, int or None")        
        
    if not all(isinstance(x,str) for x in [scale, xscale, yscale]):
        raise AliGaterError("in gateTiltedLine: ","scale, xscale and yscale must be str if specified")
    if not all(x.lower() in ['linear','logish','bilog'] for x in [scale, xscale, yscale]):
        raise AliGaterError("in gateTiltedLine: ","scale, xscale and yscale must be either of 'linear', 'logish' or 'bilog'")
    """
    The gating problem can be divided into three sections. Line is given by y=kx+m.
    ---A--- ----B---- ----C---
    Area before the tilted line(A), this exists if any value in startPoint is not None 
    
    The tilted line section (B), which is everything between startPoint[0] and endPoint[0]. Exists if startPoint[0] - endPoint[0] > 0
    
    Area after the tilted line (C), exists if the tilted line reaches either of the limits in endLim or the max/min values of the input data
    
    """
    if scale.lower() != 'linear':
        xscale = scale
        yscale = scale
    vX = getGatedVector(fcsDF, xCol, vI=vI, return_type="nparray")
    vY = getGatedVector(fcsDF, yCol, vI=vI, return_type="nparray")    
    if xscale.lower()=='logish':
        vX=logishTransform(vX, T)
    if yscale.lower()=='logish':
        vY=logishTransform(vY, T) 
        
    if xscale.lower()=='bilog':
        vX=bilogTransform(vX, T)         
    if yscale.lower()=='bilog':
        vY=bilogTransform(vY, T) 
        
    x_min = min(vX)
    x_max = max(vX)
    y_min = min(vY)
    y_max = max(vY)
    
    if startPoint[0] is not None:
        if xscale != 'linear':
            B_startx = transformWrapper(startPoint[0], T, xscale)
        else:
            B_startx = startPoint[0]
    else:
        B_startx = x_min

    #Check sign of k:
    if theta < 0:
        negative_k=True
    else:
        negative_k=False
        
    if endLimits[0] is not None:
        if endLimits[0] < startPoint[0]:
            raise AliGaterError("in gateTiltedLine: ","x-endLimit cannot be less than x-startPoint")
        if xscale != 'linear':
            B_xlim = transformWrapper(endLimits[0], T, xscale)
        else:
            B_xlim = endLimits[0]
    else: 
        B_xlim = x_max
        
    if endLimits[1] is not None:
        if yscale != 'linear':
            B_ylim = transformWrapper(endLimits[1], T, yscale)
        else:
            B_ylim = endLimits[1]  
    else:
        if not negative_k:
            B_ylim = y_max
        else:
            B_ylim = y_min
      
    if startPoint[1] is not None:
        if yscale != 'linear':
            B_starty = transformWrapper(startPoint[1], T, yscale)
        else:
            B_starty = startPoint[1]   
    else:
        B_starty = y_min
    
    if B_startx <= x_min:
        has_a = False
    else:
        has_a = True
    
    

    #Calculate y=kx+m
    rad_theta = np.radians(theta)
    #k = B_startx * np.tan(rad_theta)
    k = np.tan(rad_theta)
    m = B_starty - k*B_startx
    
    y_at_xlim = k*B_xlim + m
    #Only way to not have c is if the tilted line intersects x=xmax without hitting ymax or B_ylim
    #Initial requirement for that is that B_xlim > xmax or B_xlim = None
    #The y at x_max must then also be less than B_ylim and ymax
    if B_xlim >= x_max:
        if (y_at_xlim < B_ylim and y_at_xlim < y_max) and not negative_k:
            has_c = False
        elif (y_at_xlim > B_ylim and y_at_xlim > y_min) and negative_k:
            has_c = False
        else: 
            has_c = True
    else:
        has_c = True
    

    #Now we know if there is an A and C section but extent of B section is unknown
    x_at_ylim = (B_ylim - m)/k
    if not negative_k:
        if y_at_xlim <= B_ylim:
            B_endx = B_xlim
            B_endy = y_at_xlim
        elif x_at_ylim <= B_xlim:
            B_endx = (B_ylim - m)/k
            B_endy = B_ylim
    else:
        if y_at_xlim >= B_ylim:
            B_endx = B_xlim
            B_endy = y_at_xlim
        elif x_at_ylim <= B_xlim:
            B_endx = (B_ylim - m)/k
            B_endy = B_ylim
    
    
    result_vI = []
    if population.lower() == 'upper':
        for index, x, y in zip(vI,vX,vY):
            if x < B_startx:
                if y >= B_starty:
                    result_vI.append(index)
                    continue
            elif x >= B_startx and x <= B_endx:
                if y >= k*x+m:
                    result_vI.append(index)
                    continue
            elif x > B_endx:
                if (y >= B_ylim or y >= y_at_xlim) and not negative_k:
                    result_vI.append(index)
                    continue
                elif (y >= B_ylim and y >= y_at_xlim) and negative_k:
                    result_vI.append(index)
                    continue
            else:
                raise AliGaterError("in gateTiltedLine: ","Unhandled coordinate")
    else:
        for index, x, y in zip(vI,vX,vY):
            if x < B_startx:
                if y <= B_starty:
                    result_vI.append(index)
                    continue
            elif x >= B_startx and x < B_endx:
                if y <= k*x+m:
                    result_vI.append(index)
                    continue
            elif x >= B_endx:
                if (y <= B_ylim and y <= y_at_xlim) and not negative_k:
                    result_vI.append(index)
                    continue
                elif (y <= B_ylim or y <= y_at_xlim) and  negative_k:
                    result_vI.append(index)
                    continue
            else:
                raise AliGaterError("in gateTiltedLine: ","Unhandled coordinate")
      
    outputGate = AGgate(result_vI, parentGate, xCol, yCol, name)
    #plotting section
    if plot or filePlot is not None:
        fig,ax = plotHeatmap(fcsDF, xCol, yCol, vI, scale=scale,thresh=T)
        addLine(fig,ax, inverseTransformWrapper([x_min,B_starty], scale=scale, T=T), inverseTransformWrapper([B_startx, B_starty], scale=scale, T=T), scale=scale, T=T)
        addLine(fig,ax, inverseTransformWrapper([B_startx,B_starty], scale=scale, T=T), inverseTransformWrapper([B_endx, B_endy], scale=scale, T=T), scale=scale, T=T)
        addLine(fig,ax, inverseTransformWrapper([B_endx, B_endy], scale=scale, T=T), inverseTransformWrapper([x_max,B_endy], scale=scale, T=T), scale=scale, T=T)
        if filePlot is not None:
            plt.savefig(filePlot)
            if not plot:
                plt.close(fig)
        if plot:
            plt.show()
            plt.clf()
            plotHeatmap(fcsDF, xCol, yCol, result_vI, scale=scale, thresh=T)
            plt.show()
            
    return outputGate
