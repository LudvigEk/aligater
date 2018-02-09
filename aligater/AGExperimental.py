#!/usr/bin/env python3
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

import aligater as ag
import numpy as np
import sys

sentinel = object()
from scipy.ndimage.filters import gaussian_filter

def variableQuadGate(fcsDF, xCol, yCol, threshList, testRange, position, testSteps=20, vI=sentinel, plot=True, scale='linear', bins=300, sigma=2, T=1000, only_solution=False):
    if not isinstance(threshList, list):
        raise TypeError("threshList argument must be a list instance with [xbottom, xtop, yleft, yright] thresholds")
    if not (len(threshList)==5 or len(threshList)==4):
        raise ValueError("threshList must contain 4 thresholds; [xbottom, xtop, yleft, yright],\nor be a solution from previous quadgate; [xbottom, xtop, yleft, yright, score]")
    if not all(isinstance(i,(float,int)) for i in threshList):
        raise TypeError("ThreshList elements must be float or int")
    if not (threshList[0]==threshList[1] or threshList[2]==threshList[3]):
        raise ValueError("Invalid values in threshList, one axis must be fix.\nEither xbottom must be equal to xtop or yleft must be equal to yright")
    vX=ag.getGatedVector(fcsDF, xCol, vI)
    vY=ag.getGatedVector(fcsDF, yCol, vI)
    xscale = yscale = scale
    heatmap, xedges, yedges = ag.getHeatmap(vX, vY, bins, scale, xscale, yscale, T)
    smoothedHeatmap=gaussian_filter(heatmap.astype(float),sigma=3)
    solutions=[]
    result = ag.evaluatePartitioning(smoothedHeatmap,xedges, yedges, threshList[0], threshList[1], threshList[2], threshList[3], scale, T)
    reportStr="Input quadgate solution score: "+str(result)+"\n"
    sys.stderr.write(reportStr)
#    if plot:
#        ag.customQuadGate(fcsDF, "IgD", "CD27", vI=vI,  threshList=threshList[0:4], scale=scale, plot=plot)
    solutions.append([threshList[0], threshList[1], threshList[2], threshList[3],result])
    
    testThreshRange=np.linspace(testRange[0],testRange[1],testSteps)
    for testThresh in testThreshRange:
        if position.lower()=='left':
            result = ag.evaluatePartitioning(smoothedHeatmap,xedges, yedges, threshList[0], threshList[1], testThresh, threshList[3], scale, T)
            solutions.append([threshList[0], threshList[1],testThresh,threshList[3], result])
        elif position.lower()=='right':
            result = ag.evaluatePartitioning(smoothedHeatmap,xedges, yedges, threshList[0], threshList[1],threshList[2],testThresh, scale, T)
            solutions.append([threshList[0], threshList[1], threshList[2],testThresh, result])
        elif position.lower()=='top':
            result = ag.evaluatePartitioning(smoothedHeatmap,xedges, yedges, threshList[0],testThresh,threshList[2], threshList[3], scale, T)
            solutions.append([threshList[0],testThresh,threshList[2], threshList[3], result])    
        elif position.lower()=='bottom':
            result = ag.evaluatePartitioning(smoothedHeatmap,xedges, yedges, testThresh,threshList[1], threshList[2], threshList[3], scale, T)
            solutions.append([testThresh,threshList[1], threshList[2], threshList[3], result])  
            
    scores = [solution[4] for solution in solutions]
    solutionIndex = scores.index(min(scores))
    reportStr="Tested "+str(len(solutions)-1)+" solution(s) excluding the input solution\nBest solution had score: "+str(scores[solutionIndex])+"\n"
    sys.stderr.write(reportStr)
    
    if only_solution:
        if plot:
            ag.customQuadGate(fcsDF, "IgD", "CD27", vI=vI,  threshList=solutions[solutionIndex][:-1], scale=scale, plot=plot)
        return solutions[solutionIndex]
    
    topLeft, topRight, bottomRight, bottomLeft = ag.customQuadGate(fcsDF, "IgD", "CD27", vI=vI,  threshList=solutions[solutionIndex][:-1], scale=scale, plot=plot)
    return topLeft, topRight, bottomRight, bottomLeft, solutions[solutionIndex]

def findBin(heatmap, value, edges, scale='linear', T=1000):
    assert heatmap.shape[0] == heatmap.shape[1]
    nBins = heatmap.shape[0]
    vmin = min(edges)
    vmax = max(edges)
    if scale.lower()=='logish':
        value = ag.convertToLogishPlotCoordinate(value, vmin, vmax, T)
    binIndex = (value-vmin)/(vmax-vmin) * nBins
    binIndex = int(round(binIndex,0))
    if binIndex<0:
        binIndex=0
    if binIndex>=nBins:
        binIndex=nBins-1
    return binIndex

def evaluatePartitioning(heatmap, xedges, yedges, xB, xT, yL, yR, scale, T):
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
        rectangle = ag.heatmapRect(heatmap,xedges,yedges,xThresh,yThresh,orientation, scale,T)
        result+=ag.edgeDensity(heatmap, rectangle, orientation)
    return result

def heatmapRect(heatmap, xedges, yedges, xlim, ylim, orientation, scale='linear', T=1000):
    #SOME ERROR HANDLING NEEDED

    assert heatmap.shape[0] == heatmap.shape[1]
    nBins = heatmap.shape[0]
    xmin = min(xedges)
    xmax = max(xedges)
    ymin = min(yedges)
    ymax = max(yedges)
    if scale.lower()=='logish':
        xlim = ag.convertToLogishPlotCoordinate(xlim, xmin, xmax, T)
        ylim = ag.convertToLogishPlotCoordinate(ylim, ymin, ymax, T)
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
    edgeDensity=edgeDensity/(len(xBins)+len(yBins))
    return edgeDensity

def penalty(dx, phi):
    #penalty = pow(x,phi) - 1
    penalty = dx*phi
    #print("penalty for "+str(x)+": "+str(penalty))
    return penalty

def penaltyValleySeek(fcsDF, xCol, x0, vI=sentinel, direction='up', phi=1, sigma=3, bins=300, scale='linear', T= 1000):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        raise ValueError("Passed index contains no events")    
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if not isinstance(x0, (float,int, np.float64)):
        raise ValueError("x0 must be specified as float or integer value.\nThis is the predefined cutOff to reevaluate with penalty.")
    if type(direction) is not str:
        raise ValueError("direction must be specified as string 'up' or 'down', which direction to reevaluate threshold with penalty.")
    if not (direction=='up' or direction=='down'):
        raise ValueError("direction must be specified as string 'up' or 'down', which direction to reevaluate threshold with penalty.")
    if scale=='logish':
        smoothedHisto, binData=ag.getDensityFunc(fcsDF,xCol, vI, sigma, bins, scale='logish',T=T)
        searchSpace=ag.logishTransform(binData,T)
        x0=ag.logishTransform([x0], T)[0]
    else:
        smoothedHisto, binData=ag.getDensityFunc(fcsDF,xCol, vI, sigma, bins)
        searchSpace=binData

    interval=[]
    if direction=='up':
        interval.append(x0)
        interval.append(max(binData))
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
        raise ValueError("Specified interval is too narrow (Not enough data points to find a valley)")
    
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