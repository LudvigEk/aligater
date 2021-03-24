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

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

#AliGater imports
import aligater.AGConfig as agconf
from aligater.AGCore import customQuadGate, getDensityFunc
from aligater.AGPlotRoutines import getHeatmap, convertToLogishPlotCoordinate, logishTransform, bilogTransform, inverseLogishTransform, inverseBilogTransform, addLine, plotHeatmap, transformWrapper, inverseTransformWrapper
from aligater.AGCython import gateThreshold, gatePointList
from aligater.AGFileSystem import getGatedVector, reportGateResults, invalidAGgateParentError, invalidSampleError, filePlotError, AliGaterError, markerError
from aligater.AGClasses import AGgate, AGsample

sentinel = object()
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import halfnorm

def variableQuadGate(fcs, names, xCol, yCol, threshList, testRange, position, testSteps=20, parentGate=None, scale='linear', bins=300, sigma=2, T=1000, only_solution=False, scoreThresh=1,filePlot=None, QC=False, update=False):
    """
    A quadgate function that tests the threshold on one axis and attempts to find the least-dense point to set that limit. \n
    Does not plot by itself, it does that through a subcall to customQuadGate
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in a sample object.
    names : list-like
        | list of string with four names for the output gated populations in clockwise order from top-left;
        | Top-left, top-right, bottom-right, bottom-left
    xCol, yCol : str
        Marker labels.
    threshList : list-like of float or int
        Requires four float or int values. These are the thresholds in each direction for the gate. \n
        In order; bottom x threshold, top x threshold, left y threshold, right y threshold.
        Either the x or y thresholds must be equal (that axis will be fix).
    testRange : list-like of float or int
        Requires two float or int values, this is the interval in which the function will search for the least dense point to set the threshold.
    position : str
        Which direction to test. \n 
        Options are: 'left', 'right', 'top', 'bottom'
    testSteps, int, optional, default: 20
        The testRange will be equally divided into this many limits and each tested.\n
        Increase this to increase precision.
    parentGate : AGgate object, optional, default: None
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.        
    scale : str, optional, default: 'linear'
        If plotting enabled, which scale to be used on both axis.
    T : int, optional, default: 1000
        If plotting enabled and scale is logish, the threshold for linear-loglike transition.    
    bins : int, optional, default: 300
        Defines the resolution of the heatmap.
    sigma : float, optional, default: 2
        Smoothing factor for the heatmap.
    only_solution : bool, optional, default: False
        Changes return behaviour.\n
        If True the function does not plot and only returns the resulting float threshold.
    scoreThresh : float, optional, default: 1
        Changes the acceptance of new solutions. \n 
        A solution will be accepted if it's better when multiplied by this value. \n
        Can be both higher and lower than one.
    filePlot : str, optional, default: None
        Option to plot the gate to file to specified path.\n
        Warning: might overwrite stuff.
    update : bool, optional, default: False
        If True will add the resulting gated population(s) to the sample objects gate list in adition to returning the gate object.\n
        If False (default), returns an AGgate object without adding it to the sample object.
    QC : bool, optional, default: False
        If True, adds a downsampled image of the gating view to the gate object. These can be collected by an AGExperiment object if it's QC flag is also True.

    
    **Returns**

    AGClasses.AGgate, AGClasses.AGgate, AGClasses.AGgate, AGClasses.AGgate, float
        Returns AGClasses.AGgate objects for the four gated populations and then the treshold with the highest score.\n 
        The populations are returned in clockwise order; top-left, top-right, bottom-right, bottom-left and then the highest scoring threshold.
    float
        If only_solution is True, returns the threshold with the best score.

    .. note::
            Can be called iteratively to test many variations of the gate by setting only_solution to True. \n 
            A final call can then be made with only_solution set to False to apply the gate.
    
    **Examples**

    None currently.
    """
    
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False    
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in variableQuadGate:")
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError('in variableQuadGate:')
    else:
        vI=parentGate()
    fcsDF=fcs()
    if len(vI)<5:
        if not only_solution:
            sys.stderr.write("WARNING: in variableQuadGate: Passed population ("+str(parentGate.name)+") contains <5 events, returning empty gates and None threshold\n") 
            return AGgate([],parentGate,xCol,yCol,names[0]), AGgate([],parentGate,xCol,yCol,names[1]), AGgate([],parentGate,xCol,yCol,names[2]), AGgate([],parentGate,xCol,yCol,names[3]), None
        if only_solution:
            sys.stderr.write("WARNING: in variableQuadGate: Passed population ("+str(parentGate.name)+") contains <5 events, returning None threshold\n") 
            return None   
    if not isinstance(threshList, list):
        raise AliGaterError("threshList argument must be a python list with [xbottom, xtop, yleft, yright] (float or int) thresholds", "in variableQuadGate: ")
    if not (len(threshList)==5 or len(threshList)==4):
        raise AliGaterError("threshList must contain 4 thresholds; [xbottom, xtop, yleft, yright],\nor be a solution from previous quadgate; [xbottom, xtop, yleft, yright, score]", "in variableQuadGate: ")
    if not all(isinstance(i,(float,int)) for i in threshList):
        raise AliGaterError("ThreshList elements must be float or int", "in variableQuadGate: ")
    if not (threshList[0]==threshList[1] or threshList[2]==threshList[3]):
        raise AliGaterError("Invalid values in threshList, one axis must be fix.\nEither xbottom must be equal to xtop or yleft must be equal to yright","in variableQuadGate: ")
    vX=getGatedVector(fcsDF, xCol, vI)
    vY=getGatedVector(fcsDF, yCol, vI)
    xscale = yscale = scale
    heatmap, xedges, yedges = getHeatmap(vX, vY, bins, scale, xscale, yscale, T)
    smoothedHeatmap=gaussian_filter(heatmap.astype(float),sigma=3)
    solutions=[]
    result = evaluatePartitioning(smoothedHeatmap,xedges, yedges, threshList[0], threshList[1], threshList[2], threshList[3], scale, T)
    if agconf.ag_verbose:
        reportStr="Input quadgate solution score: "+str(result)+"\n"
        sys.stderr.write(reportStr)

    solutions.append([threshList[0], threshList[1], threshList[2], threshList[3],result])
    
    testThreshRange=np.linspace(testRange[0],testRange[1],testSteps)
    for testThresh in testThreshRange:
        if position.lower()=='left':
            result = evaluatePartitioning(smoothedHeatmap,xedges, yedges, threshList[0], threshList[1], testThresh, threshList[3], scale, T)
            solutions.append([threshList[0], threshList[1],testThresh,threshList[3], result])
        elif position.lower()=='right':
            result = evaluatePartitioning(smoothedHeatmap,xedges, yedges, threshList[0], threshList[1],threshList[2],testThresh, scale, T)
            solutions.append([threshList[0], threshList[1], threshList[2],testThresh, result])
        elif position.lower()=='top':
            result = evaluatePartitioning(smoothedHeatmap,xedges, yedges, threshList[0],testThresh,threshList[2], threshList[3], scale, T)
            solutions.append([threshList[0],testThresh,threshList[2], threshList[3], result])    
        elif position.lower()=='bottom':
            result = evaluatePartitioning(smoothedHeatmap,xedges, yedges, testThresh,threshList[1], threshList[2], threshList[3], scale, T)
            solutions.append([testThresh,threshList[1], threshList[2], threshList[3], result])  
            
    scores = [solution[4] for solution in solutions]
    solutionIndex=0
    for index, score in enumerate(scores):
        if score < scoreThresh*scores[0]:
            solutionIndex=index
    if agconf.ag_verbose:
        reportStr="Tested "+str(len(solutions)-1)+" solution(s) excluding the input solution\nBest solution had score: "+str(scores[solutionIndex])+"\n"
        sys.stderr.write(reportStr)
    
    if only_solution:
        return solutions[solutionIndex]
    
    topLeft, topRight, bottomRight, bottomLeft = customQuadGate(fcs, names=names, xCol=xCol, yCol=yCol, parentGate=parentGate,  threshList=solutions[solutionIndex][:-1], scale=scale, T=T, filePlot=filePlot)
    return topLeft, topRight, bottomRight, bottomLeft, solutions[solutionIndex]


def evaluatePartitioning(heatmap, xedges, yedges, xB, xT, yL, yR, scale, T):
    #*****************INTERNAL************
    result=0
    for orientation in ['ul','ur','br','bl']:
        if orientation=='ul':
            xThresh=xT
            yThresh=yL
        elif orientation=='ur':
            xThresh=xT
            yThresh=yR
        elif orientation=='br':
            xThresh=xB
            yThresh=yR
        elif orientation=='bl':
            xThresh=xB
            yThresh=yL
        rectangle = heatmapRect(heatmap,xedges,yedges,xThresh,yThresh,orientation, scale,T)
        result+=edgeDensity(heatmap, rectangle, orientation)
    return result

def heatmapRect(heatmap, xedges, yedges, xlim, ylim, orientation, scale='linear', T=1000):
    #*****************INTERNAL*************
    if not heatmap.shape[0] == heatmap.shape[1]:
        reportStr="passed heatmap must be symmetrical, passed heatmap dimensions: "+str(heatmap.shape[0])+str(heatmap.shape[1])
        raise AliGaterError("in heatmapRect:",reportStr)
    nBins = heatmap.shape[0]
    xmin = min(xedges)
    xmax = max(xedges)
    ymin = min(yedges)
    ymax = max(yedges)
    if scale.lower()=='logish':
        xlim = convertToLogishPlotCoordinate(xlim, xmin, xmax, T)
        ylim = convertToLogishPlotCoordinate(ylim, ymin, ymax, T)
    xBin = (xlim-xmin)/(xmax-xmin) * nBins
    xBin = int(round(xBin,0))
    if xBin<0:
        xBin=0
    if xBin>=nBins:
        xBin=nBins-1
    yBin = (ylim-ymin)/(ymax-ymin) * nBins
    yBin = int(round(yBin,0))
    if yBin<0:
        yBin=0
    if yBin>=nBins:
        yBin=nBins-1
    if orientation.lower()=='ul':
        return [[0,xBin],[yBin,nBins-1]]
    elif orientation.lower()=='ur':
        return [[xBin,nBins-1],[yBin,nBins-1]]
    elif orientation.lower()=='br':
        return [[xBin,nBins-1],[0,yBin]]
    elif orientation.lower()=='bl':
        return [[0,xBin],[0,yBin]]
    else:
        raise RuntimeError("Unhandled case, unexpected error in heatmapRect")

def edgeDensity(heatmap, rect, orientation):
    #**************INTERNAL***************
    if orientation.lower() not in ['ul','ur','br','bl']:
        raise("orientation.lower() not in ['ul','ur','br','bl']")
    edgeDensity=0.0
    xBins=np.arange(rect[0][0],rect[0][1])
    yBins=np.arange(rect[1][0],rect[1][1])
    
    if orientation.lower()=='ul':
        yEdgeBin=rect[1][0]
        xEdgeBin=rect[0][1]
    elif orientation.lower()=='ur':
        yEdgeBin=rect[1][0]
        xEdgeBin=rect[0][0]
    elif orientation.lower()=='br':
        yEdgeBin=rect[1][1]
        xEdgeBin=rect[0][0]
    elif orientation.lower()=='bl':
        yEdgeBin=rect[1][1]
        xEdgeBin=rect[0][1]
        
    for xindex in xBins:
        value=heatmap[xindex][yEdgeBin]
        edgeDensity+=value
    for yindex in yBins:
        value=heatmap[xEdgeBin][yindex]
        edgeDensity+=value
    if len(xBins)+len(yBins)==0:
        edgeDensity=0
    else:
        edgeDensity=edgeDensity/(len(xBins)+len(yBins))
    return edgeDensity

def penalty(dx, phi):
    #*********INTERNAL*************
    #Possible to change behaviour to exponential penalty, TODO: flag? should be under *args **kwargs on parents
    #penalty = pow(x,phi) - 1
    penalty = dx*phi
    return penalty

def penaltyValleySeek(fcs, xCol, x0, xEnd=None, parentGate=None, direction='up', phi=1, sigma=3, bins=300, scale='linear', T= 1000):
    """
    
    Similar to valleySeek, but searches from a starting point in a given direction with penalty.\n
    Prefers the starting position unless a new threshold is better even with penalty.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol : str
        Marker label.
    x0 : float
        Starting position for search.
    xEnd : float, optional, default: None
        Endpoint for search
    parentGate : AGgate object, optional, default: None
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    direction, str, optional, default: 'up'
        Which direction to perform search. \n
        'up' means from starting point to increasing values on the axis. \n
        'down' means from starting point towards decreasing values on the axis.
    phi : float, optional, default: 1
        Factor for the penalty function. See notes.\n
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
            The penalty function adds the absolute difference between starting coordinate and end coordinate multiplied by phi to the score of a solution. \n
            Higher phi will apply heavier penalty for increasing distance from the starting position (larger 'jumps').\n
            If phi is set very low (or 0) it's likely that the best threshold will be found at the very end of the axis.\n
    
    .. note::
            The input heatmap is smoothed to avoid finding localized minima due to noise. This has the drawback that returned thresholds are approximate of the least dense point, the error is:\n
            +- ( bin width / 2 )

    **Examples**

    None currently.
    """
    
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError("in penaltyValleySeek: ")
    else:
        vI=parentGate()
    fcsDF=fcs()
    if len(vI)==0:
        raise AliGaterError("Passed index contains no events","in penaltyValleySeek: ")
    if xCol not in fcsDF.columns:
        raise AliGaterError(str(xCol)+" not found in dataframes columns","in penaltyValleySeek: ")
    if not isinstance(x0, (float,int, np.float64)):
        raise AliGaterError("x0 must be specified as float or integer value.\nThis is the predefined cutOff to reevaluate with penalty.","in penaltyValleySeek: ")
    if type(direction) is not str:
        raise AliGaterError("Parameter direction had an invalid type, found "+str(type(direction))+" expected "+str(type(str)),"in penaltyValleySeek: ")
    if not (direction=='up' or direction=='down'):
        raise AliGaterError("direction must be specified as either 'up' or 'down', which direction to reevaluate threshold with penalty.","in penaltyValleySeek: ")
    if xEnd is not None:
        if not isinstance(xEnd,(float, int)):
            raise AliGaterError("in penaltyValleySeek: ", "xEnd had unexpected dtype, expected float/int, found: "+str(type(xEnd)))
    if scale=='logish':
        smoothedHisto, binData=getDensityFunc(fcsDF,xCol, vI, sigma, bins, scale='logish',T=T)
        searchSpace=logishTransform(binData,T)
        x0=logishTransform([x0], T)[0]
        if xEnd is not None:
            xEnd=logishTransform([xEnd], T)[0]
    else:
        smoothedHisto, binData=getDensityFunc(fcsDF,xCol, vI, sigma, bins)
        searchSpace=binData
        
    if xEnd is not None:
        end_binData=xEnd
        
    interval=[]
    if direction=='up':
        interval.append(x0)
        if xEnd is not None:
            if end_binData <= x0:
                raise
            interval.append(end_binData)
        else:
            interval.append(max(binData))
    else:
        if xEnd is not None:
            if end_binData >= x0:
                raise
            interval.append(end_binData)
        else:
            interval.append(min(binData))
        interval.append(x0)
    assert len(interval)==2

    vIndicies=[]
    for index, x in np.ndenumerate(searchSpace[:-1]):
        #Note the non-inclusive upper bound, critical to stay in-bound of array index
        if x >= interval[0] and x < interval[1]:
            vIndicies.append(index[0])
            
    if len(vIndicies)<=3:
        sys.stderr.write("in penaltyValleySeek: Specified interval is too narrow (Not enough data points to find a valley), returning x0\n")
        return x0
    
    minVal=np.inf
    minValIndex=0
  

    for index in vIndicies:
        x=smoothedHisto[index]/len(vI)
        xData=searchSpace[index]
        penalizedX=x+penalty(abs(xData-x0), phi)
        if penalizedX < minVal:
            minVal=penalizedX
            minValIndex=index
    return (binData[minValIndex+1]+binData[minValIndex])/2

def densityDelimitation(fcs, xCol, parentGate=None, interval=['start','end'], sigma=3, bins=300, limit_threshold=0.05, direction='left', scale='linear', T= 1000):
    """
    Function that finds a threshold where the density is some fraction of the maximum density in the specified interval. 
    Returns the rough coordinate where this occurs.
    
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
    limit_threshold : float, optional, default: 0.05
        Fraction based threshold relative to the maximum density in the given interval.
    direction : str, optional, default : 'right'
        The search direction in the density function.
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

    if type(interval[0]) is str:
        if interval[0].lower() in ['start', 'first']:
            interval[0]=min(vX) 
        else:
            raise AliGaterError("in valleySeek: ","limit specified as string but option unrecognized, expected 'first' or 'start', found "+interval[0].lower())
    if type(interval[1]) is str:
        if interval[1].lower() in ['end', 'last']:
            interval[1]=max(vX) 
        else:
            raise AliGaterError("in valleySeek: ","limit specified as string but option unrecognized, expected 'last' or 'end', found "+interval[1].lower())

    final_vX=[]
    for x in vX:
        if x<interval[1] and x>interval[0]:
            final_vX.append(x)
            
    vX=np.asarray(final_vX)
    
    
    if scale.lower()!='linear':
        vX = transformWrapper(vX, scale=scale, T=T)
        interval[1]=transformWrapper([interval[1]],scale=scale, T=T)[0]
        interval[0]=transformWrapper([interval[0]],scale=scale, T=T)[0]
    
    
    histo, binData = np.histogram(vX,bins=bins)
    smoothedHisto=gaussian_filter1d(histo.astype(float),sigma, mode='nearest')
    
    maximum_density=max(smoothedHisto)
    maximum_density_index=np.argmax(smoothedHisto)

    if isinstance(maximum_density_index,(list, np.ndarray)):
        sys.stderr.write("WARNING, in densityDelimitation: Multiple maxima.\n")
        if direction.lower()=='right':
            maximum_density_index=int(maximum_density_index[-1])
        elif direction.lower()=='left':
            maximum_density_index=int(maximum_density_index[0])
            
    density_threshold=limit_threshold*maximum_density

    if direction.lower()=='left':
        endPoint=0
        searchStep=-1
    elif direction.lower()=='right':
        endPoint=len(smoothedHisto)
        searchStep=1
    else:
        raise AliGaterError("in densityDelimitation","direction must be specified as 'right' or 'left'.")
        
    #Search from maximum density bin, in search direction (i.e. left or right)
    for i in np.arange(maximum_density_index, endPoint, searchStep):
        if smoothedHisto[i] < density_threshold:
            threshold_index = i
            break
    else:
        return np.inf
    
    if direction.lower()=='left':
        result=(binData[threshold_index]+binData[threshold_index+1])/2
    if direction.lower()=='right':
        result=(binData[threshold_index]+binData[threshold_index-1])/2
    
    if scale.lower()!='linear':
            result = inverseTransformWrapper([result],scale=scale, T=T)[0]
    return result



def halfNormalDistribution(fcs, xCol, mean, direction, parentGate=None, bins=300, scale='linear',T=1000):
    """
    Estimates a normal distribution in an axis by only evaluating one direction from a given mean.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol, yCol : str
        Marker labels.
    mean : float
        Mean for the normal distribution that should be estimated.
    parentGate : AGgate object, optional, default: None
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    bins : int, optional, default: 300
        Defines the resolution of the density histogram from which the normal distribution is estimated.
    scale : str, optional, default: 'linear'
        Which scale to be used on axis. \n
        **WARNING**: in contrast to many other functions, this actually affects more than plotting behaviour. See notes!
    T : int, optional, default: 1000
        If scale is logish, the threshold for linear-loglike transition.
        
    .. note::
            If a scale is changed from the default 'linear', the normal distribution is estimated on the transformed values (i.e. what you would see if plotting with this scale)\n
            **The returned values will then also be the mean and sigma of the transformed values**.\n 
            To reverse transform see aligater.AGPlotRoutines.inverseTransformWrapper.
            When setting a treshold based on these values (such as mean+2*sigma), use transformed values and then invert.
        
    **Returns**

    float, float
        Mean, standard deviation

    **Examples**

    None currently.
    """
    if not isinstance(direction,str):
        raise AliGaterError("Parameter direction had an invalid type, found "+str(type(direction))+" expected "+str(type(str)),"in halfNormalDistribution: ")
    if direction.lower() not in ["left","right"]:
        raise AliGaterError("Specify direction as 'left' or 'right'","in halfNormalDistribution: ")
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in halfNormalDistribution: ")
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError("in halfNormalDistribution: ")
    else:
        vI=parentGate()
    fcsDF=fcs()
    if len(vI)<5:
        sys.stderr.write("WARNING, in halfNormalDistribution: Passed index contains too few events, returning input mean and standard deviation zero\n")  
        return mean, 0.0
    if xCol not in fcsDF.columns:
        raise AliGaterError(str(xCol)+" not found in dataframes columns","in halfNormalDistribution: ")
        
    data=getGatedVector(fcsDF,xCol, vI, return_type="nparray")
    distribution = []
    if direction.lower()=='right':
        for x in data:
            if x >= mean:
                distribution.append(x)
    else:
        for x in data:
            if x <= mean:
                distribution.append(x)


    if scale.lower()=='logish':
        distribution=list(logishTransform(distribution,T))
        mean=logishTransform([mean],T)[0] 
    
    if scale.lower()=='bilog':
        distribution=list(bilogTransform(distribution,T))
        mean=bilogTransform([mean],T)[0] 
    
    sumVar=0
    n=len(distribution)
    for x in range(0,n):
        sumVar += (distribution[x] - mean)**2
        
    if n != 0:    
        sigma=np.sqrt(sumVar/n)
    else:
        sigma=0

        
    return mean, sigma

def dijkstraStep(heatmap, xBin, yBin, bins):
    #*************INTERNAL************
    if yBin==bins-1:
        if heatmap[xBin-1,yBin] < heatmap[xBin-1,yBin-1]:
            return 0
        else: 
            return -1
    elif yBin==0:
        if heatmap[xBin-1,yBin] < heatmap[xBin-1,yBin+1]:
            return 0
        else: 
            return 1
    else:
        steps=[heatmap[xBin-1,yBin-1],heatmap[xBin-1,yBin],heatmap[xBin-1,yBin+1]]
        step=steps.index(min(steps))
        return step-1
    

def shortestPath(fcsDF, xCol, yCol, boundaries, vI=sentinel,maxStep=30, sigma=3, points=5, scale='linear', xscale='linear',yscale='linear',bins=300, T=1000):
    #****************INTERNAL*******************
    #TODO: Deprecated, slow, use the cython implementation
    sys.stderr.write("WARNING, in shortestPath: This function is deprecated and the cython version (shortestPathMatrix) is vastly superior\n")
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    originalvI=vI
    #shortestPath estimates a stepwise shortest path through a heatmap from start to end
    #by one dimensional valleyseeking in a set of points(default 5) between those two coordinates
    #i.e. it's a type of 1d gradient descent
    #should be faster than full blown dijkstra's or 2d gradient descent
    tmpvI=gateThreshold(fcsDF,xCol, yCol,thresh=boundaries[0], orientation='horizontal', population='upper',scale=scale, vI=vI,info=False)
    vI=gateThreshold(fcsDF,xCol,yCol, thresh=boundaries[1], orientation='horizontal',population='lower',scale=scale,vI=tmpvI, info=False)
    avgBinDepth=len(vI)/(bins*bins)

    vX=getGatedVector(fcsDF, xCol, vI)
    vY=getGatedVector(fcsDF, yCol, vI)

    xscale = yscale = scale
    heatmap, xedges, yedges = getHeatmap(vX, vY, bins, scale, xscale, yscale, T)
    smoothedHeatmap=gaussian_filter(heatmap.astype(float),sigma=sigma)
    

    #Set all positions except desired end point bin to inf in the last bin-row
    #smoothedHeatmap[-1,bins-1]=0
    #print(smoothedHeatmap)
    if maxStep>bins:
        maxStep=int(np.round(bins/2))
    
    maxStep=maxStep
    #Paths is a list of score ('time spent traveling') plus a list of previous steps in that path
    #I.e. paths= [pathScore,[firstStep,secondStep...,latestStep]]
    #That means paths[0] gives us the currently fastest path (best score)
    #and paths[0][1] is it's list of steps taken; the fastests paths [step1, step2,...,lateststep]
    paths=[]
    #Set up first iteration, all paths have to take their first step from 0,0
    #So we force this
    for tmpBin in np.arange(0,maxStep,1):
        penalty=sum([smoothedHeatmap[1][skippedRow] for skippedRow in np.arange(0,maxStep,1)])

        startScore=smoothedHeatmap[1,tmpBin]+smoothedHeatmap[0,0]+penalty
        paths.append([startScore, [0,tmpBin]])

        
    paths=sorted(paths, key=lambda x: x[0])

    while len(paths[0][1]) < bins:
        #Which xBin is the current path at? Or, how many steps have this path taken?
        currentPathX=len(paths[0][1])
        #What was the last yBin of this path?
        currentPathLeftY=paths[0][1][-1]
        #What are the avaible next steps for this path?
        #We require it to only move in one direction, 'upwards' or horizontally
        #We also take into consideration the max amount of steps allowed for a jump
        maxAllowedStep=currentPathLeftY+maxStep
        if maxAllowedStep>bins:
            maxAllowedStep=bins
        currentPathAvailableSteps = [smoothedHeatmap[currentPathX,y] for y in np.arange(currentPathLeftY,maxAllowedStep,1)]
        #Adjust with penalty for jumping
        for x in np.arange(0,len(currentPathAvailableSteps),1):
            if x<2:
                penalty=0
            else:
                penalty=avgBinDepth#min(currentPathAvailableSteps[0:x])
            currentPathAvailableSteps[x]+=penalty
        #Fill the other elements (invalid steps) with infinite, this way we can figure out index
        currentPathAllY=[np.inf for y in np.arange(0,currentPathLeftY,1)]
        currentPathAllY.extend(currentPathAvailableSteps)
        tmp=[np.inf for y in np.arange(maxAllowedStep,bins,1)]
        currentPathAllY.extend(tmp)

        #Find score for best step
        score=paths[0][0]+min(currentPathAvailableSteps)
        #Find bin of best step
        step=currentPathAllY.index(min(currentPathAvailableSteps))
        
            
        #Add this step to this path
        paths[0][1].append(step)
        paths[0][0]=score
        
        #Resort the list
        paths=sorted(paths, key=lambda x: x[0])
        

    if plot:
        heatmap=np.ma.masked_where(smoothedHeatmap == 0, smoothedHeatmap)
        plt.clf()
        fig, ax = plt.subplots()
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower',aspect='equal')
        plt.xlabel(xCol)
        plt.ylabel(yCol)
        cmap=plt.get_cmap()
        cmap.set_bad(color='white')
        
    #draw line of shortest path
    count=0
    vPL=[]
    for index,step in enumerate(paths[0][1]):
        if count==0:
            previousCoord=[xedges[count],yedges[index]]
            count+=1
            continue
        coord=[xedges[count],yedges[step]]
        vPL.append(coord)
        #plt.plot(previousCoord, coord, 'k-', lw=2)
        if plot:
            fig,ax = addLine(fig,ax,previousCoord,coord)
        previousCoord=coord
        count+=1
        
    if plot:
        plt.show()
    
    #Gate on originalvI
    vOut=gatePointList(fcsDF,xCol,yCol,vPL, vI=originalvI)
    reportGateResults(originalvI,vOut)
    if plot:
        plt.clf()
        plotHeatmap(fcsDF, xCol, yCol, vOut, scale=scale)
        plt.show()
    return vOut


def gateBezier(fcs, xCol, yCol, name, parentGate=None, points=None, xParam=0, yParam=0, population='upper', startExtension='left', endExtension='right', scale='linear', xscale='linear', yscale='linear', T=1000, filePlot=None):
    """
    
    Gates the population by drawing a quadratic Bézier curve between two points with tunable parameters.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in a sample object.
    xCol,yCol : str
        Marker labels.
    points : list-like of tuple
        The points defining start and end of the bezier curve (a list of two tuples).
    name : str
        Name of the resulting gated population.
    parentGate : AGgate object, optional
        Parent population to apply the gating to. 
        If no AGgate object is passed gating is applied to the ungated data frame.
    xParam : float or list-like
        Parameter defining the x-coordinate of the midpoint in the quadratic Bézier curve.\n
        Should be a floating point value between which will be translated into a scaled coordinate between the start- and endpoints x coordinates.\n
        I.e. 0.0 means the midpoints x coordinate is the same as the starting x-coordinate and 1.0 gives a midpoint x-coordinate equal to the endpoint.\n
        If multiple sets of start and end points have been passed this argument expects a list with xParams for each Bézier curve in order.
    yParam : float or list-like
        Parameter defining the y-coordinate of the midpoint in the quadratic Bézier curve.\n
        Should be a floating point value between which will be translated into a scaled coordinate between the start- and endpoints y coordinates.\n
        I.e. 0.0 means the midpoints y coordinate is the same as the starting y-coordinate and 1.0 gives a midpoint y-coordinate equal to the endpoint.\n
        If multiple sets of start and end points have been passed this argument expects a list with yParams for each Bézier curve in order.        
    population, str, optional, default: 'upper'
        This parameter determines which population should be returned.\n
        'upper' means any events with a value above the tresholds are returned.\n
        'lower' means any events with a value below the tresholds will be returned.\n
        The default setting means the population that's considered 'positive' in flow cytometry terms is returned.
    startExtension : str, optional, default: 'left'
        If the start point is somewhere in the middle of the gating view, this parameter defines how events falling outside the extend of the Bézier curve are treated. \n
        Options are 'up', 'down', 'left' and will extend the line in the specified direction
    endExtension : str, optional, default: 'right'
        If the end point is somewhere in the middle of the gating view, this parameter defines how events falling outside the extend of the Bézier curve are treated. \n
        Options are 'up', 'down', 'left' and will extend the line in the specified direction
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
        raise invalidAGgateParentError("in gateBezier: ")
    else:
        vI=parentGate()
    if not isinstance(fcs, AGsample):
        raise invalidSampleError("in gateBezier: ")
    else:
        fcsDF=fcs()
    if len(vI)<5:
        sys.stderr.write("WARNING: in gateBezier: Passed population ("+str(parentGate.name)+") contains <5 events, returning empty gate\n") 
        return AGgate([],parentGate,xCol,yCol,name)
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise TypeError("If plotting to file is requested filePlot must be string filename")
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels") 

    if not all([isinstance(x, tuple) for x in points]):
        raise AliGaterError("in gateBezier: ","points must be list-like of tuples")
    if not all([isinstance(x, (float,int)) for y in points for x in y]):
        raise AliGaterError("in gateBezier: ","tuples in points must be float or int")
        
    if not all(isinstance(x,str) for x in [scale, xscale, yscale]):
        raise AliGaterError("in gateBezier: ","scale, xscale and yscale must be str if specified")
    if not all(x.lower() in ['linear','logish','bilog'] for x in [scale, xscale, yscale]):
        raise AliGaterError("in gateBezier: ","scale, xscale and yscale must be either of 'linear', 'logish' or 'bilog'")
    
    if not len(points) % 2 == 0:
        raise AliGaterError("in gateBezier: ","points must contain even sets of points for Bezier curves (2,4,6 etc)")

    
    if scale.lower() != 'linear':
        xscale = scale
        yscale = scale
    vX = getGatedVector(fcsDF, xCol, vI=vI, return_type="nparray")
    vY = getGatedVector(fcsDF, yCol, vI=vI, return_type="nparray")    
    if xscale.lower()!='linear':
        vX=transformWrapper(vX, T=T, scale=xscale)
    if yscale.lower()!='linear':
        vY=transformWrapper(vY, T=T, scale=yscale)
        
    x_min = min(vX)
    x_max = max(vX)
    y_min = min(vY)
    y_max = max(vY)
    

    if xscale != 'linear':
        x_coords = transformWrapper([x[0] for x in points], T, xscale)
    else:
        x_coords=[x[0] for x in points]
    if yscale != 'linear':
        y_coords = transformWrapper([y[1] for y in points], T, yscale)
    else:
        y_coords = [y[1] for y in points]
    
        
    def rect(x1, y1, x2, y2):
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1
        return (a, b)

    def beizer(xstart,ystart, xend, yend, xParam, yParam):
        xmid = xstart+abs(xstart-xend)*xParam
        ymid = ystart+abs(ystart-yend)*yParam
        (x1, y1, x2, y2) = (xstart, ystart, xmid, ymid)
        (a1, b1) = rect(xstart, ystart, xmid, ymid)
        (a2, b2) = rect(xmid, ymid, xend, yend)
        beizerPoints = []
    
        for i in range(0, 1000):
            if x1 == x2:
                continue
            else:
                (a, b) = rect(x1, y1, x2, y2)
            x = i*(x2 - x1)/1000 + x1
            y = a*x + b
            beizerPoints.append((x,y))
            x1 += (xmid - xstart)/1000
            y1 = a1*x1 + b1
            x2 += (xend - xmid)/1000
            y2 = a2*x2 + b2
        return beizerPoints

    
    #nOfBeizerCurves = len(points)/2
    
    beizerPoints=beizer(x_coords[0],y_coords[0],x_coords[1],y_coords[1],xParam[0],yParam[0])
    
    if endExtension.lower()=='up':
        endCoord = (x_coords[-1],y_max)
    elif endExtension.lower()=='right':
        endCoord = (x_max,y_coords[-1])
    elif endExtension.lower()=='down':
        endCoord = (x_coords[-1],y_min)
    else:
        raise

    if startExtension.lower()=='up':
        startCoord = (x_coords[0],y_max)
    elif startExtension.lower()=='left':
        startCoord = (x_min,y_coords[0])
    elif startExtension.lower()=='down':
        startCoord = (x_coords[0],y_min)
    else:
        raise
    
    t_vPL = [startCoord] + beizerPoints + [endCoord]
    vPL=[]
    for point in t_vPL:
        if scale.lower() != 'linear':
            x=inverseTransformWrapper(point[0], scale=scale, T=T)
            y=inverseTransformWrapper(point[1], scale=scale, T=T)
        else:
            x=point[0]
            y=point[1]
        vPL.append((x,y))
    
    result_vI=gatePointList(fcsDF,xCol,yCol,vPL, population=population, vI=parentGate())
    
    outputGate = AGgate(result_vI, parentGate, xCol, yCol, name)
    
    
    #plotting section
    if plot or filePlot is not None:
        fig,ax = plotHeatmap(fcsDF, xCol, yCol, vI, scale=scale,thresh=T)
        addLine(fig,ax, vPL[0], vPL[1], scale=scale, T=T)
        addLine(fig,ax, vPL[-2],vPL[-1], scale=scale, T=T)
        for i in np.arange(0,len(beizerPoints),2):
            x1=inverseTransformWrapper(beizerPoints[i][0], scale=scale, T=T)
            x2=inverseTransformWrapper(beizerPoints[i+1][0], scale=scale, T=T)
            y1=inverseTransformWrapper(beizerPoints[i][1], scale=scale, T=T)
            y2=inverseTransformWrapper(beizerPoints[i+1][1], scale=scale, T=T)
            addLine(fig,ax, [x1,y1], [x2,y2], scale=scale, T=T)

        if filePlot is not None:
            plt.savefig(filePlot)
            if not plot:
                plt.close(fig)
        if plot:
            plt.show()
            plt.close()
            plotHeatmap(fcsDF, xCol, yCol, result_vI, scale=scale, thresh=T)
            plt.show()
            
    return outputGate
