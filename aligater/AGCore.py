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
from aligater.AGPlotRoutines import plotHeatmap, plot_gmm, addLine, addAxLine, convertToLogishPlotCoordinates, convertToBiLogPlotCoordinates, logishBin, logishTransform, inverseLogishTransform
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

def gmm2D(fcs, xCol, yCol, nOfComponents, parentGate=None, *args, **kwargs):
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
        
    if not isinstance(fcs, AGsample):
        raise invalidSampleError("in gmm2D: ")
    else:
        fcsDF=fcs()
    
    vX = getGatedVector(fcsDF,xCol,vI=vI,return_type="nparray")
    vY = getGatedVector(fcsDF,yCol,vI=vI,return_type="nparray")
    fcsArray=np.array([vX,vY]).T
    gmm = GaussianMixture(n_components=nOfComponents,*args,**kwargs)
    gmm.fit(fcsArray)
    return gmm

def gateGMM(fcs, name, xCol, yCol, gmm, parentGate=None, sigma=1, update=False, QC=False):
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
        fig,ax = plotHeatmap(fcsDF, xCol, yCol, vI, aspect='equal')
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
        angle=ellipses[3]
        vTmp = gateEllipsoid(fcsDF, xCol, yCol, xCenter, yCenter, width/2, height/2, np.radians(angle), vI, info=False)
        vOutput.extend(vTmp)
    #The result is all unique events that was inside any ellipse
    vResult=list(set(vOutput))

    reportGateResults(vI, vResult)
    outputGate=AGgate(vResult, parentGate, xCol, yCol, name)
    if plot:
        fig, ax = plotHeatmap(fcsDF, xCol, yCol, vResult)
    return outputGate
    
    

def getPCs(fcsDF, xCol, yCol, centerCoord=None, vI=None):
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
    if not len(vX)==len(vY):
        raise AliGaterError("Unequal amount of data points for "+str(xCol)+" and "+str(yCol),"in getPCs(internal): ")
    
    if bManualCenter:
        meanX=centerCoord[0]
        meanY=centerCoord[1]
    else:
        meanX=np.mean(vX)
        meanY=np.mean(vY)
    
    #Center dataset around the centroid or a custom center    
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

def getHighestDensityPoint(fcs, xCol, yCol, parentGate=None, bins=300):
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
    vX=getGatedVector(fcsDF, xCol, vI)
    vY=getGatedVector(fcsDF, yCol, vI)
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
    


def gatePC(fcs, xCol, yCol, name, parentGate=None, widthScale=1, heightScale=1, center='centroid', customCenter=None, filePlot=None, update=False, QC=False, **kwargs):
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
    update : bool, optional, default: False
        If True will add the resulting gated population(s) to the sample objects gate list in adition to returning the gate object.\n
        If False (default), returns an AGgate object without adding it to the sample object.
    QC : bool, optional, default: False
        If True, adds a downsampled image of the gating view to the gate object. These can be collected by an AGExperiment object if it's QC flag is also True.
        
    **Returns**

    AGClasses.AGsample object    
        if update is True (default) 
    AGClasses.AGgate object
        if update is False

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
        center=getHighestDensityPoint(fcs, xCol, yCol, parentGate)

    elif center.lower() == 'centroid':
        center=None
    else:
        center=customCenter
        
    if plot or filePlot is not None:
        fig, ax = plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')

    center, eigen1, eigen2 = getPCs(fcsDF, xCol, yCol, center, vI)
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
       

    result=gateEllipsoid(fcsDF, xCol, yCol,xCenter=center[0],yCenter=center[1], majorAxis=[eigen1[1],eigen1[2]],majorRadii=width ,minorAxis=[eigen2[1],eigen2[2]], minorRadii=height,vI=vI)

    if plot or filePlot is not None:
        #addLine(fig, ax, center, PC1)
        #addLine(fig, ax, center, PC2)
        ax.add_patch(Ellipse(center, 2*width, 2*height, np.degrees(angle),fill=False,edgecolor='#FF0000', linestyle='dashed'))
        if filePlot is not None:
            plt.savefig(filePlot)
            if not plot:
                plt.close(fig)
        if plot:
            plt.show()
            plotHeatmap(fcsDF, xCol, yCol,result)
            plt.show()
            plt.clf()
    if parentGate is not None:
        outputGate=AGgate(result, parentGate, xCol, yCol, name)
    else:
        outputGate=AGgate(result, fcs.full_index(), xCol, yCol, name)
    if update:
        fcs.update(outputGate, QC=QC)
    return outputGate

def getVectorCoordiantes(length, angle):
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
    if scale=='logish':
        BinEdges=logishBin(data,bins,T)
        histo = np.histogram(data, BinEdges)
    else:    
        histo=np.histogram(data, bins)
    smoothedHisto=gaussian_filter1d(histo[0].astype(float),sigma)

    return smoothedHisto, histo[1]

def valleySeek(fcs, xCol, parentGate=None, interval=['start','end'], sigma=3, bins=300, scale='linear', T= 1000):
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
    
    if scale=='logish':
        smoothedHisto, binData=getDensityFunc(fcsDF,xCol, vI, sigma, bins, scale='logish',T=T)
    else:
        smoothedHisto, binData=getDensityFunc(fcsDF,xCol, vI, sigma, bins)
    
    
    if type(interval[0]) is str:
        if interval[0].lower() in ['start', 'first']:
            interval[0]=min(binData) 
        else:
            raise AliGaterError("in valleySeek: ","limit specified as string but option unrecognized, expected 'first' or 'start', found "+interval[0].lower())
    if type(interval[1]) is str:
        if interval[1].lower() in ['end', 'last']:
            interval[1]=max(binData) 
        else:
            raise AliGaterError("in valleySeek: ","limit specified as string but option unrecognized, expected 'last' or 'end', found "+interval[1].lower())
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
        return (interval[0]+interval[1])/2
    minVal=np.inf
    minValIndex=0
    for index in vIndicies:
        if smoothedHisto[index] < minVal:
            minVal=smoothedHisto[index]
            minValIndex=index

    result=(binData[minValIndex+1]+binData[minValIndex])/2
    return result
            
def quadGate(fcs, names, xCol, yCol, xThresh, yThresh, parentGate=None, scale='linear',T=1000, update=False, QC=False):
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
    
    update : bool, optional, default: False
        If True will add the resulting gated population(s) to the sample objects gate list in adition to returning the gate object.\n
        If False (default), returns an AGgate object without adding it to the sample object.
        
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
    
    if plot:
        if scale=='logish':
            fig,ax=plotHeatmap(fcsDF, xCol, yCol,vI,aspect='auto', scale=scale, thresh=T)    
        else:
            fig, ax = plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')
        addAxLine(fig,ax,xThresh,'vertical',scale=scale, T=T)
        addAxLine(fig,ax,yThresh,'horisontal',scale=scale, T=T)
        plt.show()    
        
    TopLeft=AGgate(vTopLeft, parentGate, xCol, yCol, names[0])
    TopRight=AGgate(vTopRight, parentGate, xCol, yCol, names[1])
    BottomRight=AGgate(vBottomRight, parentGate, xCol, yCol, names[2])
    BottomLeft=AGgate(vBottomLeft, parentGate, xCol, yCol, names[3])
    if agconf.ag_verbose:
        reportStr="quadGate results in clockwise order from top-left: "+str(len(vTopLeft))+", "+str(len(vTopRight))+", "+str(len(vBottomRight))+", "+str(len(vBottomLeft))+"\n"
        sys.stderr.write(reportStr)
    if update:
        if QC:
            if agconf.ag_verbose:
                sys.stderr.write("QC requested on quadgate. Only saving image for: "+str(names[0]))+"\n"
        fcs.update(TopLeft, QC=QC)
        fcs.update(TopRight, QC=False)
        fcs.update(BottomRight, QC=False)
        fcs.update(BottomLeft, QC=False)
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
        return 0,0,0
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if scale.lower()=='linear':
        x=getGatedVector(fcsDF,xCol, vI, return_type="nparray")
    elif scale.lower()=='logish':
        x=getGatedVector(fcsDF,xCol, vI, return_type="nparray")
        x=logishTransform(x,T)
    
    histo=np.histogram(x, bins)
        
    mean=np.mean(x)
    sigma=np.std(x)
    maxIndex=np.argmax(histo[0])
    
    if isinstance(maxIndex, np.ndarray):
        maxVal=(histo[1][maxIndex[0]]+histo[1][maxIndex[0]+1])/2
    else:
        maxVal=(histo[1][maxIndex]+histo[1][maxIndex+1])/2
    
    if scale.lower()=='logish':
        result=inverseLogishTransform([mean, sigma, maxVal],T)
        mean=result[0]
        sigma=abs(result[1])
        maxVal=result[2]
        
    return mean, sigma, maxVal

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


def customQuadGate(fcs, names, xCol, yCol,threshList, parentGate=None, scale='linear',T=1000, filePlot=None, update=False, QC=False):
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
    update : bool, optional, default: False
        If True will add the resulting gated population(s) to the sample objects gate list in adition to returning the gate object.\n
        If False (default), returns an AGgate object without adding it to the sample object.  
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
    if update:
        if QC:
            if agconf.ag_verbose:
                sys.stderr.write("QC requested on customQuadGate. Only saving image for: "+str(names[0]))+"\n"
        fcs.update(TopLeft, QC=QC)
        fcs.update(TopRight, QC=False)
        fcs.update(BottomRight, QC=False)
        fcs.update(BottomLeft, QC=False)
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
    return None
