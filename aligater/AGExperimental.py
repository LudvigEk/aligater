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
import numpy as np
import sys

sentinel = object()
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import halfnorm

def variableQuadGate(fcs, xCol, yCol, threshList, testRange, position, testSteps=20, parentGate=sentinel, scale='linear', bins=300, sigma=2, T=1000, only_solution=False, scoreThresh=1,filePlot=None):
    """
    A quadgate function that tests the threshold on one axis and attempts to find the least-dense point to set that limit. \n
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
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
    parentGate : AGgate object, optional
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
        
    **Returns**

    List-like, List-like, List-like, List-like, float
        If only_solution is False, returns the four gated population as list-like indicies in clockwise order;\n
        top left, top right, bottom right, bottom right \n
        It also returns the threshold it found to be best.
    float
        If only_solution is True, returns the threshold with the best score.

    **Notes**    
    
    Can be called iteratively to test many variations of the gate by setting only_solution to True. \n 
    A final call can then be made with this set to True or used to call customQuadGate.\n
    
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
    if len(vI)<ag.minCells:
        sys.stderr.write("Passed index contains no events\n") 
        if only_solution:
            return []
        else:
            return [],[],[],[],[]    
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
    solutionIndex=0
    for index, score in enumerate(scores):
        if score < scoreThresh*scores[0]:
            solutionIndex=index
    reportStr="Tested "+str(len(solutions)-1)+" solution(s) excluding the input solution\nBest solution had score: "+str(scores[solutionIndex])+"\n"
    sys.stderr.write(reportStr)
    
    if only_solution:
        return solutions[solutionIndex]
    
    topLeft, topRight, bottomRight, bottomLeft = ag.customQuadGate(fcsDF, xCol, yCol, vI=vI,  threshList=solutions[solutionIndex][:-1], scale=scale, filePlot=filePlot)
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
    """
    
    Similar to valleySeek, but searches from a starting point in a given direction with penalty.\n
    Prefers the starting position unless a new threshold is better even with penalty.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol : str
        Marker label.
    x0 : float
        Starting position for search.
    vI : list-like or AGgate object
        Parent population to apply the gating to.
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

    **Notes**
    
    The penalty function adds the absolute difference between starting coordinate and end coordinate multiplied by phi to the score of a solution. \n
    Higher phi will apply heavier penalty for the distance from the starting position.\n
    If phi is set very low (or 0) it's likely that the best threshold will be found at the very end of the axis.\n
    
    Note that the input heatmap is smoothed to avoid finding localized minima due to noise. This has the drawback that returned thresholds are approximate of the least dense point, the error is:\n
    +- ( bin width / 2 )

    **Examples**

    None currently.
    """
    
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

def halfNormalDistribution(fcsDF, xCol, mean, direction, vI=sentinel,bins=300, scale='linear',T=1000):
    if direction.lower() not in ["left","right"]:
        sys.stderr.write("Specify direction as 'left' or 'right'")
        return None
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return 0,0
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    data=ag.getGatedVector(fcsDF,xCol, vI, return_type="nparray")
    distribution = []
    if direction.lower()=='up':
        for x in data:
            if x >= mean:
                distribution.append(x)
    else:
        for x in data:
            if x <= mean:
                distribution.append(x)


    if scale.lower()=='logish':
        distribution=list(ag.logishTransform(distribution,T))
        mean=ag.logishTransform([mean],T)[0] 
        
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
    if ag.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    originalvI=vI
    #shortest path estimates a stepwise shortest path through a heatmap from start to end
    #by one dimensional valleyseeking in a set of points(default 5) between those two coordinates
    #i.e. 1d gradient descent
    #should be faster than full blown dijkstra's or 2d gradient descent
    tmpvI=ag.gateThreshold(fcsDF,xCol, yCol,thresh=boundaries[0], orientation='horisontal', population='upper',scale=scale, vI=vI,info=False)
    vI=ag.gateThreshold(fcsDF,xCol,yCol, thresh=boundaries[1], orientation='horisontal',population='lower',scale=scale,vI=tmpvI, info=False)
    avgBinDepth=len(vI)/(bins*bins)
#    print("jump penalty: "+str(avgBinDepth))
    vX=ag.getGatedVector(fcsDF, xCol, vI)
    vY=ag.getGatedVector(fcsDF, yCol, vI)
    xscale = yscale = scale
    heatmap, xedges, yedges = ag.getHeatmap(vX, vY, bins, scale, xscale, yscale, T)
    smoothedHeatmap=gaussian_filter(heatmap.astype(float),sigma=sigma)
    
    #smoothedHeatmap

    #Set all positions except desired end point bin to inf in the last bin-row
    #smoothedHeatmap[-1,bins-1]=0
    #print(smoothedHeatmap)
    if maxStep>bins:
        maxStep=int(np.round(bins/2))
    
    maxStep=maxStep
    #Paths is a list of score ('time spent traveling') plus a list previous steps in that path
    #I.e. paths= [pathScore,[firstStep,secondStep...,latestStep]]
    #That means paths[0] gives us the currently fastest path
    #and paths[0][1] it's list of steps taken; the fastests paths [step1, step2,...,lateststep]
    paths=[]
    #Set up first iteration, all paths have to take their first step from 0,0
    #So we force this
    for tmpBin in np.arange(0,maxStep,1):
        penalty=sum([smoothedHeatmap[1][skippedRow] for skippedRow in np.arange(0,maxStep,1)])
        #print(penalty)
        startScore=smoothedHeatmap[1,tmpBin]+smoothedHeatmap[0,0]+penalty
        paths.append([startScore, [0,tmpBin]])
        #print(str(startScore)+"\n")
        
    paths=sorted(paths, key=lambda x: x[0])
    #print("initial, sorted, paths")
    #print(paths)
    #print("\n")
    #print(bins)
    while len(paths[0][1]) < bins:
        #Which xBin is the current path at? Or, how many steps have this path taken?
        currentPathX=len(paths[0][1])
        #What was the last yBin of this path?
        currentPathLeftY=paths[0][1][-1]
        #What are the avaible next steps for this path?
        #We require it to only move in one direction, 'upwards' or horisontally
        #We also defined a max amount of steps to jump
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

        #score for best step
        score=paths[0][0]+min(currentPathAvailableSteps)
        #bin of best step
        step=currentPathAllY.index(min(currentPathAvailableSteps))
        
        #It's possible that in the last step the algorithm cannot reach the only 'allowed' jump
        #This allows that jump for the very last bin
#        if step==np.inf:
#            step=bins-1
            
        #add this step to this path
        paths[0][1].append(step)
        paths[0][0]=score
        
        #Resort the list
        paths=sorted(paths, key=lambda x: x[0])
        
    #print("\nbest path:")
    #print(paths[0][1])
    if plot:
        heatmap=np.ma.masked_where(smoothedHeatmap == 0, smoothedHeatmap)
        ag.plt.clf()
        fig, ax = ag.plt.subplots()
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ag.plt.imshow(heatmap.T, extent=extent, origin='lower',aspect='equal')
        ag.plt.xlabel(xCol)
        ag.plt.ylabel(yCol)
        cmap=ag.plt.get_cmap()
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
        #ag.plt.plot(previousCoord, coord, 'k-', lw=2)
        if plot:
            fig,ax = ag.addLine(fig,ax,previousCoord,coord)
        previousCoord=coord
        count+=1
        
    if plot:
        ag.plt.show()
    
    #Gate on originalvI
    vOut=ag.gatePointList(fcsDF,xCol,yCol,vPL, vI=originalvI)
    ag.reportGateResults(originalvI,vOut)
    if plot:
        ag.plt.clf()
        ag.plotHeatmap(fcsDF, xCol, yCol, vOut, scale=scale)
        ag.plt.show()
    return vOut



