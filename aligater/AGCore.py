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
        #If not available, default to terminal
        return 'terminal'

def gmm2D(fcsDF, xCol, yCol, nOfComponents, vI=sentinel, *args, **kwargs):
    """
    Fits a scikit.learn GaussianMixture object to the data and returns it
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol, yCol : str
        Marker labels.
    nOfComponents : int
        Number of components to use in the mixture model.
    vI : list-like or AGgate object
        Parent population to apply the gating to.
    args, kwargs : 
        optional arguments to pass on to GaussianMixture.

    **Returns**

    GaussianMixture object fit to the data.

    **Examples**

    None currently.
    """
    if vI is sentinel:
        vI=fcsDF.index
    #randomly_sampled = np.random.choice(vI, size=len(vI), replace=False)
    vX = getGatedVector(fcsDF,xCol,vI=vI,return_type="nparray")
    vY = getGatedVector(fcsDF,yCol,vI=vI,return_type="nparray")
    fcsArray=np.array([vX,vY]).T
    gmm = GaussianMixture(n_components=nOfComponents,*args,**kwargs)
    gmm.fit(fcsArray)
    return gmm

def gateGMM(fcsDF,xCol, yCol, gmm, vI=sentinel, sigma=1):
    """
    Function that can interpret and gate data based on a GaussianMixture object from sklearn.mixture
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol, yCol : str
        Marker labels.
    vI : list-like or AGgate object
        Parent population to apply the gating to.
    sigma : float, optional, default: 1
        Number of standard deviations to scale the mixture model with.

    **Returns**

    AGClasses.AGgate object

    **Examples**

    None currently.
    """
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

    for ellipses in vEllipses:
        xCenter=ellipses[0][0]
        yCenter=ellipses[0][1]
        width=ellipses[1]
        height=ellipses[2]
        angle=ellipses[3]
        vTmp = ag.gateEllipsoid(fcsDF, xCol, yCol, xCenter, yCenter, width/2, height/2, np.radians(angle), vI, info=False)
        vOutput.extend(vTmp)

    vResult=list(set(vOutput))

    ag.reportGateResults(vI, vResult)
    if plot:
        fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol, vResult)
    return vOutput
    
def getGatedVector(fcsDF, gate, vI=sentinel, return_type="pdseries", dtype=np.float64):
    """
    Collects list-like of measured intensities for a population. 
    Also see getGatedVectors
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    gate : str
        Marker label.
    vI : list-like or AGgate object
        Population to collect.
    return_type : str, optional, default: "pdseries"
        Format of returned list-like, options are: "pdseries" "nparray"
    dtype : Type, optional, default: numpy.float64
        Data type of values in the returned listlike

    **Returns**

    List-like

    **Examples**

    None currently.
    """
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
    """
    Collects list-like of measured intensities for a population. \n
    Useful to collect both intensity coordinates for events in a view.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. 
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    gate1, gate2 : str
        Marker labels.
    vI : list-like or AGgate object
        Population to collect.
    return_type : str, optional, default: "pdseries"
        Format of returned list-like, options are: "pdseries" "nparray", matrix
    dtype : Type, optional, default: numpy.float64
        Data type of values in the returned list-like

    **Returns**
    
    numpy.array[numpy.array, numpy.array]
        If return type is 'nparray'
        numpy array of arrays in order: x-array, y-array
    numpy.array[list]
        If return_type is 'matrix' returns a numpy array with list-likes of two; x-coord, y-coord
    list-like, list-like
        if return_type is 'pdseries' returns two pandas.Series objects in order; X, Y

    **Examples**

    None currently.
    """
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
    
#TODO: Keep non-cython version?
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

def getHighestDensityPoint(fcs, xCol, yCol, parentGate=sentinel, bins=300):
    """
    Returns coordinates for the point in the view with the highest number of events.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
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
    if parentGate is sentinel:
        vI=fcs.full_index()
    elif not parentGate.__class__.__name__ == "AGgate":
        raise TypeError("invalid AGgate object")
    else:
        vI=parentGate()
    fcsDF=fcs()
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
    if ag.ag_verbose:
        reportString="After gating, "+str(len(vOutput))+" out of "+str(len(vI))+" events remain.\n"
        sys.stderr.write(reportString)
    return


def gatePC(fcs, xCol, yCol, name, parentGate=sentinel, widthScale=1, heightScale=1, center='centroid', customCenter=None, filePlot=None, update=True):
    """
    Function that performs a 2D principal component analysis and gates an ellipse based on the results.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol, yCol : str
        Marker labels.
    name : str
        Name to the resulting gated population.
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
    update : bool, optional, default: True
        Changes the return behaviour. If True (default) will apply the gate and add a new gate object to the samples
        gate list then return the updated AGsample.\n
        If false, returns an AGgate object.

    **Returns**

    AGClasses.AGsample object    
        if update is True (default) 
    AGClasses.AGgate object
        if update is False

    **Examples**

    None currently.
    """
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")
    if parentGate is sentinel:
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
    if len(vI)==0:
        sys.stderr.write("Passed index contains no events") 
        return []
    if center.lower() == 'density':
        center=ag.getHighestDensityPoint(fcs, xCol, yCol, parentGate)

    elif center.lower() == 'centroid':
        center=None
    else:
        center=customCenter
        
    if plot or filePlot is not None:
        fig, ax = ag.plotHeatmap(fcsDF, xCol, yCol,vI,aspect='equal')

    center, eigen1, eigen2 = ag.getPCs(fcsDF, xCol, yCol, center, vI)
    center, PC1, PC2 = ag.getPCSemiAxis(center, eigen1, eigen2, widthScale, heightScale)
    width=ag.getVectorLength(center, PC1)
    height=ag.getVectorLength(center, PC2)
    angle=ag.calculateAngle(center, PC1)

    result=ag.gateEllipsoid(fcsDF, xCol, yCol,xCenter=center[0],yCenter=center[1], majorAxis=[eigen1[1],eigen1[2]], minorAxis=[eigen2[1],eigen2[2]],majorRadii=width, minorRadii=height,vI=vI)

    if plot or filePlot is not None:
        ag.addLine(fig, ax, center, PC1)
        ag.addLine(fig, ax, center, PC2)
        ax.add_patch(ag.Ellipse(center, 2*width, 2*height, np.degrees(angle),fill=False,edgecolor='#FF0000', linestyle='dashed'))
        if filePlot is not None:
            ag.plt.savefig(filePlot)
            if not plot:
                ag.plt.close(fig)
        if plot:
            ag.plt.show()
            ag.plotHeatmap(fcsDF, xCol, yCol,result)
            ag.plt.show()
            ag.plt.clf()
    if parentGate is not sentinel:
        outputGate=ag.AGgate(result, parentGate, xCol, yCol, name)
    else:
        outputGate=ag.AGgate(result, fcs.full_index(), xCol, yCol, name)
    if update:
        fcs.update(outputGate, QC=True)
    return outputGate

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
    """
    Function finds the least dense point in a given interval for values in a channel.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol : str
        Marker label.
    name : str
        Name to the resulting gated population.
    vI : list-like or AGgate object
        Parent population to apply the gating to.
    interval : list-like, optional, default: ['start','end']
        Interval to limit the search, defaults to entire axis.
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

    **Examples**

    None currently.
    """
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
    """
    Function that gates four populations from one view by drawing a cross.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n 
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol, yCol : str
        Marker labels.
        
    xThresh : float
        Threshold for vertical line.

    yThresh : float
        Threshold for horisontal line.

    vI : list-like or AGgate object
        Parent population to apply the gating to.
        
    scale : str, optional, default: 'linear'
        If plotting enabled, which scale to be used on both axis.
        
    T : int, optional, default: 1000
        If plotting enabled and scale is logish, the threshold for linear-loglike transition

    **Returns**

    list-like, list-like, list-like, list-like
        Returns list-like indicies for the four gated populations.\n 
        In clockwise order; top-left, top-right, bottom-right, bottom-left    

    **Examples**

    None currently.
    """
    
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

def gateCorner(fcs, name, xCol, yCol, xThresh, yThresh, xOrientation='upper', yOrientation='upper', Outer=False, parentGate=sentinel, bins=300, scale='linear', T=1000, update=True, filePlot=None):
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
    update : bool, optional, default: True
        Changes the return behaviour. If True (default) will apply the gate and add a new gate object to the samples
        gate list then return the updated AGsample.
        If false, returns an AGgate object.

    **Returns**

    AGClasses.AGsample object    
        if update is True (default) 
    AGClasses.AGgate object
        if update is False

    **Examples**

    None currently.
    """    
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False    
    if parentGate is sentinel:
        vI=fcs.full_index()
    elif not parentGate.__class__.__name__ == "AGgate":
        raise TypeError("invalid AGgate object")
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
    if not Outer:
        vOutput=list(set(vOutput).intersection(vI))
    else:
        vOutput=list(set(vI).difference(vOutput))
    
    if len(vOutput)==0:
        return []
    
    if plot or filePlot is not None:
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
        if filePlot is not None:
            ag.plt.savefig(filePlot)
            if not plot:
                ag.plt.close(fig)
        if plot:
            plt.show()
            plt.clf()
            ag.plotHeatmap(fcsDF, xCol, yCol, vOutput, scale=scale)
            plt.show()

        
    if parentGate is not sentinel:
        outputGate=ag.AGgate(vOutput, parentGate, xCol, yCol, name)
    else:
        outputGate=ag.AGgate(vOutput, fcs.full_index(), xCol, yCol, name)
    if update:
        fcs.update(outputGate, QC=True)
    reportGateResults(vI, vOutput)
    return outputGate


def customQuadGate(fcsDF, xCol, yCol,threshList, vI=sentinel, scale='linear',T=1000, filePlot=None):
    """
    A quadgate function with one axis fix and the other variable. The threshList argument decides which axis is fix and which can vary.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. 
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol, yCol : str
        Marker labels.
    vI : list-like or AGgate object
        Parent population to apply the gating to.
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

    List-like, list-like, list-like, list-like
        Indicies of events in each quadrant, in order: topleft, topright, bottomright, bottomleft
        
    **Examples**

    None currently.
    """     
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return []
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
    if plot or filePlot is not None:
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
        if filePlot is not None:
            ag.plt.savefig(filePlot)
            ag.plt.close()
        if plot:
            ag.plt.show()       
    return vTopLeft, vTopRight, vBottomRight, vBottomLeft

#TODO
def backGate(fcsDF, xCol, yCol, vI=sentinel, backPop=sentinel, markersize=2, scale='linear',xscale='linear',yscale='linear',T=1000, filePlot=None):
    """
    Highlights a population onto another view/population.\n
    Typically used to see where rarer populations are located on an earlier view in a gating strategy.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol : str
        Marker label.
    backPop : list-like or AGgate object
        Background population.
    vI : list-like or AGgate object
        Population that should be highlighted.
    markersize : float, optional, default: 2
        Size of events of the 'backPop' population.
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
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")    
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
    if filePlot is not None:
        ag.plt.savefig(filePlot)
        ag.plt.close()
    return None
