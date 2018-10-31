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
 
#import aligater as ag
import sys
import numpy as np
from cpython cimport bool
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

#AliGater imports
import aligater.AGConfig as agconf
from aligater.AGFileSystem import getGatedVector, reportGateResults, invalidAGgateParentError, invalidSampleError, filePlotError, AliGaterError, markerError
from aligater.AGPlotRoutines import addLine, addAxLine, getHeatmap, logishTransform, plotHeatmap, plot_densityFunc, convertToLogishPlotCoordinate
from aligater.AGClasses import AGsample,AGgate

#For cython optimisations with boundscheck and wraparound
cimport cython
#For cython optimization with numpy arrays
cimport numpy as np
#Declare some types for cython
ctypedef np.float64_t dtype_t
ctypedef np.int32_t itype_t
# declare interface to C code functions
cdef extern from "AGc.h":
    void c_Stat_GetMeanAndVariance_double(const double* aData, const int nSize, double &mean, double &var)
    void c_multiply (double* array, double value, int m, int n)

#
#Dummy object for python handling
sentinel=object()

@cython.boundscheck(False)
def gateEllipsoid(fcsDF, str xCol, str yCol, float xCenter, float yCenter, list majorAxis, float majorRadii, list minorAxis, float minorRadii, vI=sentinel, str population="inner", bool info=True):   
    #******************Internal*******************
    #******Should have wrapper********
    """
    Function that gates an ellipsoid given by direction and length of its axis'.
    Currently internal function, will produce a user friendly(ier) wrapper.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame. \n
        If data is stored in an AGSample object this can be retrieved by
        calling the sample, i.e. mysample().
    xCol, yCol : str
        Marker labels.
    xCenter, yCenter : float
        Center coordinates of ellipsoid.
    majorAxis : List-like
        List-like normalized coordinate direction of major axis; [x,y]
    minorAxis : List-like
        List-like normalized coordinate direction of minor axis; [x,y]
    majorRadii : float
        Length of major axis.
    minorRadii : float
        Length of minor axis.        
    vI : list-like
        Parent population to apply the gating to.
    population : str, optional, default: 'inner'
        Options: 'inner', 'outer'\n
        Which population should be returned, inside or outside the ellipse.
    info : bool, optional, default: True
        Should gating results be reported (as std err)

    **Returns**

    List-like
        Index of resulting population.

    **Examples**

    None currently.
    """    
    if not xCol in fcsDF.columns:
        raise NameError("xCol not in passed dataframes' columns")
    if not yCol in fcsDF.columns:
        raise NameError("yCol not in passed dataframes' columns")
    if xCol==yCol:
        raise NameError("xCol and yCol cannot be the same")
    if vI is sentinel:
        vI=fcsDF.index  
    if len(vI)==0:
        sys.stderr.write("Passed index contains no events") 
        return []
    if population.lower()=="inner":
        lower=True
    elif population.lower()=="outer":
        lower=False
    else:
        raise TypeError("Specify desired population, 'outer' or 'inner' in regard to ellipsoid")    
    vOutput=[]

    cdef np.ndarray[dtype_t, ndim = 1] vX = getGatedVector(fcsDF, xCol, vI, return_type='nparray',dtype=np.float64)
    cdef np.ndarray[dtype_t, ndim = 1] vY = getGatedVector(fcsDF, yCol, vI, return_type='nparray', dtype=np.float64)
    cdef float Ux=majorAxis[0]*1/majorRadii
    cdef float Uy=majorAxis[1]*1/majorRadii
    cdef float Vx=minorAxis[0]*1/minorRadii
    cdef float Vy=minorAxis[1]*1/minorRadii
    cdef float x, y, e0, e1, result
    cdef int n = len(vI)
    cdef int i, index
    cdef np.ndarray[itype_t, ndim = 1] arrIdx = np.asarray(vI, dtype=np.int32)
    cdef np.ndarray[itype_t, ndim = 1] vOutTmp = np.zeros_like(vI, dtype=np.int32)
    cdef int nOfResults=0
    
    assert len(vX)==len(vY)==len(arrIdx)
    for i in range(n):
        x=vX[i]
        y=vY[i]
        index=arrIdx[i]
        x=x-xCenter
        y=y-yCenter
        e0= x*Ux+y*Uy;
        e1= x*Vx+y*Vy;
        result = e0*e0+e1*e1; 
        if lower:
            if result  <= 1:
                vOutTmp[nOfResults]=index
                nOfResults+=1
        elif result  >= 1:
            vOutTmp[nOfResults]=index
            nOfResults+=1
            
    vOutput=vOutTmp[0:nOfResults].tolist()
    if (len(vOutput) == 0 and population.lower() == "inner"):
        sys.stderr.write("No events inside ellipsoid\n")
    if (len(vOutput) == 0 and population.lower() == "outer"):
        sys.stderr.write("No events outside ellipsoid\n")
    if info:
        reportGateResults(vI, vOutput)
    return vOutput

@cython.wraparound(False)
@cython.boundscheck(False)
def gateThreshold(fcs, str name, str xCol, yCol=None, thresh=None, orientation="vertical", parentGate=None, population="upper", scale='linear', T=1000, update=False,filePlot=None, QC=False, *args, **kwargs):
    """
    Threshold gating function. Can be called with one or two markers which affects plotting.\n
    Call with two markers to get a two dimensional view or with one col which will instead show a density plot.
    
    **Parameters**
    
    fcs : AGClasses.AGSample object
        Flow data loaded in an sample object.
    xCol : str
        Marker label.
    yCol : str, optional, default: None
        Marker label. If specified, opts for a two dimensional view when plotting. Otherwise, a density plot.
    thresh : float
        Threshold for cutoff, see orientation and population parameters for how this is applied.
    orientation, str, optional, default: 'vertical'
        If two markers are passed this parameter decies if the cut is made on the x- or y-axis.\n
        'vertical' corresponds to drawing a vertical line in the plot, i.e. cut-off is applied to the x-axis.
        'horisontal' corresponds to drawing a horisontal line in the plot, i.e. cut-off is applied to the y-axis.
    population, str, optional, default: 'upper'
        This parameter determines which population should be returned.\n
        'upper' means any events with a value above the treshold are returned.\n
        'lower' means any events with a value below the treshold will be returned.\n
        The default value means the population that's considered 'positive' in flow cytometry terms is returned.
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
    filePlot : str, optional, default: None
        Option to plot the gate to file to specified path. \n
        Warning: might overwrite stuff.    
    QC : bool, optional, default: False
        If True, adds a downsampled image of the gating view to the gate object. These can be collected by an AGExperiment object if it's QC flag is also True.
        
    **Returns**

    AGClasses.AGgate object
         Returns an AGgate object for the gated population.

    **Examples**

    None currently.
    """ 
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    if not isinstance(fcs,AGsample):
        raise invalidSampleError("in gateThreshold:")
    if parentGate is None:
        vI=fcs.full_index()
    elif not isinstance(parentGate,AGgate):
        raise invalidAGgateParentError('in gateThreshold:')
    else:
        vI=parentGate()
    fcsDF=fcs()
    if filePlot is not None:
        if not isinstance(filePlot,str):
            raise filePlotError("in gateThreshold:")
    if not isinstance(thresh, (float,int)):
        raise AliGaterError("thresh must be specified as float or integer value","in gateThreshold: ")
    if yCol is not None:
        if yCol not in fcsDF.columns:
            raise markerError("in gateThreshold, marker: '"+str(yCol)+"'")
    if xCol not in fcsDF.columns:
        raise markerError("in gateThreshold, marker: '"+str(yCol)+"'")
    if population.lower() not in ["upper","lower"]:
        raise AliGaterError("Specify desired population, 'upper' or 'lower' in regard to set threshold","in gateThreshold: ")
    if orientation.lower() not in ["horisontal","vertical"]:
        raise AliGaterError("Specify desired population, 'upper' or 'lower' in regard to set threshold","in gateThreshold: ") 
    if len(vI)<5:
        sys.stderr.write("WARNING, in gateThreshold: Passed parent population to "+str(name)+" contains too few events, returning empty gate.\n") 
        outputGate=AGgate([],parentGate,xCol,yCol,name)
        if update:
            fcs.update(outputGate, QC=QC)         
        return outputGate

    if yCol is None:
        densityPlot=True
    else:
        densityPlot=False

    cdef list vOutput=[]
    cdef int index, i
    cdef dtype_t value
    cdef np.ndarray[itype_t, ndim = 1] tmp_vI = np.asarray(vI,dtype=np.int32)
    cdef np.ndarray[dtype_t, ndim = 1] data
    cdef int nOfEvents = len(vI)
    cdef dtype_t t = np.float64(thresh)
    if orientation.lower() == "vertical":
        data=getGatedVector(fcsDF, xCol, vI, return_type="nparray", dtype=np.float64)
        if population.lower()=="upper":
            for i in range(0,nOfEvents,1):
                index=tmp_vI[i]
                value = data[i]
                if value>=t:
                    vOutput.append(index)
        else:
            for i in range(0,nOfEvents,1):
                index=tmp_vI[i]
                value = data[i]
                if value < t:
                    vOutput.append(index)
    if yCol is not None:
        if orientation.lower() == "horisontal":
            data=getGatedVector(fcsDF, yCol, vI, return_type="nparray", dtype=np.float64)
            if population.lower()=="upper":
                for i in range(0,nOfEvents,1):
                    index=tmp_vI[i]
                    value = data[i]
                    if value >= t:
                        vOutput.append(index)
            else:
                for i in range(0,nOfEvents,1):
                    index=tmp_vI[i]
                    value = data[i]
                    if value < t:
                        vOutput.append(index)

    if (plot or filePlot is not None) and not densityPlot:
        fig,ax = plotHeatmap(fcsDF, xCol, yCol, vI, scale=scale,thresh=T)
        addAxLine(fig,ax,thresh,orientation,scale=scale, T=T)
        if filePlot is not None:
            plt.savefig(filePlot)
            if not plot:
                plt.close(fig)
        if plot:
            plt.show()
            plt.clf()
            plotHeatmap(fcsDF, xCol, yCol, vOutput, scale=scale,thresh=T)
            plt.show()
    if (plot or filePlot is not None) and densityPlot:
        fig,ax =plot_densityFunc(fcsDF,xCol, vI, scale=scale, T=T,*args,**kwargs)
        addAxLine(fig,ax,thresh,orientation,scale=scale, T=T)
        if filePlot is not None:
            plt.savefig(filePlot)
            if not plot:
                plt.close(fig)
        if plot:
            plt.show()
    if agconf.ag_verbose:
        reportGateResults(vI, vOutput)
    

    outputGate=AGgate(vOutput, parentGate, xCol, yCol, name)
    
    if update:
        fcs.update(outputGate, QC=QC)
    return outputGate


@cython.boundscheck(False)
@cython.wraparound(False)
def shortestPathMatrix(fcs, str name, str xCol, str yCol, list boundaries, parentGate=None, float sigma=3, int maxStep=20, str scale='linear', str xscale='linear', str yscale='linear', int bins=300, float T=1000, update=True, QC=False):
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    cdef list vI, originalvI
    if not fcs.__class__.__name__=="AGsample":
        raise TypeError("invalid AGsample object")
    if parentGate is None:
        vI=fcs.full_index()
    elif not parentGate.__class__.__name__ == "AGgate":
        raise TypeError("Parent population in "+name+" is an invalid AGgate object.")
    else:
        vI=parentGate()
    if len(vI)<5:
        sys.stderr.write("Passed parent population to "+name+" contains too few events, returning empty gate.\n") 
        outputGate=AGgate([],parentGate,xCol,yCol,name)
        if update:
            fcs.update(outputGate, QC=QC)         
        return outputGate
        
    originalvI=vI
    fcsDF=fcs()
    tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=boundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=parentGate,info=False, update=False)
    vI=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=boundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=tmpvI, info=False,update=False)()
    cdef np.ndarray vX=getGatedVector(fcsDF, xCol, vI, return_type="nparray")
    cdef np.ndarray vY=getGatedVector(fcsDF, yCol, vI, return_type="nparray")
    xscale = yscale = scale
    heatmap, xedges, yedges = getHeatmap(vX, vY, bins, scale, xscale, yscale, T)
    cdef np.ndarray[dtype_t, ndim=2] smoothedHeatmap = gaussian_filter(heatmap.astype(float),sigma=sigma)
    cdef np.ndarray[dtype_t, ndim=2] cost = np.empty_like(smoothedHeatmap, dtype=np.float_)
    cdef np.ndarray[dtype_t, ndim=2] leftBin = np.empty_like(smoothedHeatmap, dtype=np.float_)
    cost.fill(np.inf)
    leftBin.fill(0)
    cost[0][0]=smoothedHeatmap[0][0]
    cost[bins-1][bins-1]=smoothedHeatmap[bins-1][bins-1]
        
    cdef int y, x
    cdef int previousX, adjustedMaxStep, dist, stepIndex
    cdef float stepCost, penalty
    cdef int leftBinIndex
    cdef np.ndarray[dtype_t, ndim=1] stepCosts = np.zeros([maxStep],dtype=np.float_)
    cdef int costIdx, tmpIdx, minCostIdx
    cdef float minCost
    for y in range(1,bins,1):
        for x in range(0,bins,1): 
            for tmpIdx in range(0,maxStep,1):
                stepCosts[tmpIdx]=10000000000000000.0
            if x-maxStep<0:
                adjustedMaxStep=x
            else:
                adjustedMaxStep=maxStep-1
            costIdx=0

            for previousX in range(x-adjustedMaxStep,x+1,1):
                dist=x-previousX
                penalty=dist*dist
                stepCost = cost[y-1,previousX]+smoothedHeatmap[y,x]+penalty
                stepCosts[costIdx]=stepCost
                costIdx+=1

            minCost=10000000000000000.0

            for tmpIdx in range(0,maxStep,1):
                if stepCosts[tmpIdx]<minCost:
                    minCost=stepCosts[tmpIdx]
                    minCostIdx=tmpIdx
            cost[y,x]=minCost
            leftBinIndex=minCostIdx+x-adjustedMaxStep
            leftBin[y,x]=leftBinIndex

    #traverse cost matrix, /w monothony
    leftBinIndex=bins-1
    if leftBinIndex > bins-1:
        leftBinIndex=bins-1

    cdef list path=[]
    for x in range(bins-1,-1,-1):
        path.append([xedges[x],yedges[leftBinIndex]])
        leftBinIndex=int(leftBin[x][leftBinIndex])
    
    path=path[::-1]

    cdef int count=0
    cdef list coord
    if plot:
        heatmap=np.ma.masked_where(smoothedHeatmap == 0, smoothedHeatmap)
        plt.clf()
        fig, ax = plt.subplots()
        extent = [xedges[0], xedges[bins], yedges[0], yedges[bins]]
        plt.imshow(heatmap.T, extent=extent, origin='lower',aspect='auto')
        plt.xlabel(xCol)
        plt.ylabel(yCol)
        cmap=plt.get_cmap()
        cmap.set_bad(color='white')
    
        #draw line of shortest path
        for coord in path:
            if count==0:
                previousCoord=coord
                count+=1
                continue 
            fig,ax = addLine(fig,ax,previousCoord,coord)
            previousCoord=coord
            count+=1
        plt.show()
    vOut=gatePointList(fcsDF,xCol,yCol,path, vI=originalvI)
    reportGateResults(originalvI,vOut)
    
    if plot:
        plt.clf()
        plotHeatmap(fcsDF, xCol, yCol, vOut, scale=scale)
        plt.show()
    if parentGate is not sentinel:
        outputGate=AGgate(vOut, parentGate, xCol, yCol, name)
    else:
        outputGate=AGgate(vOut, fcs.full_index(), xCol, yCol, name)
    if update:
        fcs.update(outputGate, QC=True)
        
    return outputGate


def gatePointList(fcsDF, xCol, yCol, vPL, population='lower',vI=sentinel, scale='linear', T=1000):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return []
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if population.lower()=='lower':
        lower=True
    elif population.lower()=='upper':
        lower=False
    else:
        raise("specify population as 'lower' or 'upper'\n")
    if scale.lower() not in ['linear', 'logish']:
        raise("specify scale as 'logish' or 'linear'\n")
    vOut=[]
    vX=getGatedVector(fcsDF, xCol, vI,return_type='nparray')
    vY=getGatedVector(fcsDF, yCol, vI,return_type='nparray')
    if scale.lower()=="logish":
        vX = logishTransform(vX,T)

    cdef float x0=min(vX)
    cdef float x1=max(vX)
    cdef int nOfBins=len(vPL)
    cdef float x, y, ylim
    cdef int index
    
    for x, y, index in zip(vX, vY, vI):
        dist = round( (x-x0)/(x1-x0) * nOfBins)
        if dist>=nOfBins:
            dist=nOfBins-1

        ylim=vPL[int(dist)][1]

        if y<ylim:
            if lower:
                vOut.append(index)
        if y>=ylim:
            if not lower:
                vOut.append(index)

    return vOut

def gateTwoPointList(fcsDF, xCol, yCol, vPL, vPL2, vI=sentinel, scale='linear', T=1000):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events\n") 
        return []
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    if scale.lower() not in ['linear', 'logish']:
        raise("specify scale as 'logish' or 'linear'\n")
    if len(vPL) != len(vPL2):
        raise("Coordinate lists must be of equal length")
    vOut=[]
    vX=getGatedVector(fcsDF, xCol, vI,return_type='nparray')
    vY=getGatedVector(fcsDF, yCol, vI,return_type='nparray')
    if scale.lower()=="logish":
        vX = logishTransform(vX,T)

    x0=min(vX)
    x1=max(vX)
    nOfBins=len(vPL)
    for x, y, index in zip(vX, vY, vI):
        dist = (x-x0)/(x1-x0) * nOfBins
        dist=int(round(dist))
        if dist>=nOfBins:
            dist=nOfBins-1

        ylimUpper=vPL2[dist][1]
        ylimLower=vPL[dist][1]

        if y<ylimUpper and y>=ylimLower:
            vOut.append(index)
            
    return vOut

@cython.boundscheck(False)
@cython.wraparound(False)
def triGate(fcsDF, str xCol, str yCol, list centerCoord, vI=sentinel, int bins=300, float T=1000, str scale='linear', int sigma=3):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)<50000:
        sys.stderr.write("Passed index contains no events\n") 
        return [],[],[]
    if xCol not in fcsDF.columns or yCol not in fcsDF.columns:
        raise TypeError("Specified gate(s) not in dataframe, check spelling or control your dataframe.columns labels")
    xscale=yscale=scale
    cdef np.ndarray[dtype_t, ndim=1] vX = getGatedVector(fcsDF, xCol, vI, return_type='nparray', dtype=np.float64)
    cdef np.ndarray[dtype_t, ndim=1] vY = getGatedVector(fcsDF, yCol, vI, return_type='nparray', dtype=np.float64)
    heatmap, xedges, yedges = getHeatmap(vX, vY, bins, scale, xscale, yscale, T)
    cdef np.ndarray[dtype_t, ndim=2] smoothedHeatmap = gaussian_filter(heatmap.astype(np.float64),sigma=sigma)
    cdef itype_t xCenterbin = np.int32(findBin(smoothedHeatmap, centerCoord[0], xedges, scale=scale, T=T))
    cdef itype_t yCenterbin = np.int32(findBin(smoothedHeatmap, centerCoord[1], yedges, scale=scale, T=T))

    cdef dtype_t bottomDens
    cdef np.ndarray[dtype_t,ndim=1] density=np.zeros(90)
    cdef itype_t adjustIdx
    cdef np.ndarray[itype_t, ndim=2] topLine, bottomLine, leftLine
    for adjustIdx in range(0,90,1):
        bottomDens, leftLine, topLine, bottomLine = triGateIteration(smoothedHeatmap, xCenterbin+adjustIdx, yCenterbin, bins)
        density[adjustIdx]=bottomDens
        
    cdef float minDens=1000000000000000000000.0
    cdef int minDensIndex
    for i in range(0,45,1):
        if density[i]<minDens:
            minDens=density[i]
            minDensIndex=i

    cdef dtype_t[:,:,:] result
    cdef np.ndarray[itype_t, ndim=2] bottomLimit, topLimit
    bottomLimit= np.append(leftLine,bottomLine,axis=0)
    topLimit= np.append(leftLine,topLine,axis=0)

    top, bottom, right = cythonGateTwoPointList(vI, vX, vY, xedges, yedges, bottomLimit, topLimit, bins, scale, T)

    return top, bottom, right

@cython.boundscheck(False)
@cython.wraparound(False)
def cythonGateTwoPointList(list vI, np.ndarray[dtype_t, ndim=1] vX, np.ndarray[dtype_t, ndim=1] vY, np.ndarray xedges, np.ndarray yedges, np.ndarray[itype_t, ndim=2] bottomlimit, np.ndarray[itype_t, ndim=2] toplimit, int bins, str scale, float T):
    if scale.lower()=="logish":
        vX = logishTransform(vX,T)
    cdef list top=[]
    cdef list bottom=[]
    cdef list right=[]
    cdef dtype_t x0 = np.float64(min(vX))
    cdef dtype_t x1 = np.float64(max(vX))
    
    cdef float iterX, iterY 
    cdef int index
    cdef itype_t i
    cdef itype_t nOfEvents = np.int32(len(vI))
    cdef int dist
    cdef float ylimUpper, ylimLower

    cdef int yUpperIndex, yLowerIndex
    assert len(vI)==len(vX)==len(vY)
    for i in range(0,nOfEvents,1):
        iterX=vX[i]
        iterY=vY[i]
        index=vI[i]
        dist = int(round((iterX-x0)/(x1-x0) * bins))
        if dist>=bins:
            dist=bins-1
        if dist < 0:
            dist=0
        yUpperIndex=toplimit[dist,1]
        yLowerIndex=bottomlimit[dist,1]
        ylimUpper=yedges[yUpperIndex]
        ylimLower=yedges[yLowerIndex]

        if ylimUpper==ylimLower:
            if iterY<ylimLower:
                bottom.append(index)
            if iterY>=ylimUpper:
                top.append(index)
        else:
            if iterY<ylimUpper and iterY>=ylimLower:
                right.append(index)
            elif iterY<ylimLower:
                bottom.append(index)
            elif iterY>=ylimUpper:
                top.append(index)
                
    return top, bottom, right


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef triGateIteration(np.ndarray[dtype_t, ndim=2] smoothedHeatmap,  itype_t xCenterbin, itype_t yCenterbin, int bins):
    cdef np.ndarray[itype_t, ndim=2] topLine, bottomLine
    cdef np.ndarray[itype_t, ndim=2] leftLine = np.zeros((xCenterbin+1,2),dtype=np.int32)
    cdef itype_t i, rightSideLength
    cdef itype_t yTopbin, yBottombin
    cdef dtype_t bottomDens
    rightSideLength = bins-xCenterbin-1 
    for i in range(xCenterbin,-1,-1):
            leftLine[i,0]=i
            leftLine[i,1]=yCenterbin
        
    #Right side (the two separating lines) should go from xCenterbin to bins-1
    #Three lines are created, leftLine from xCenterbin to 0, 
    #topLine from xCenterbin+1 that goes in a top right direction
    #vis-a-vis for bottomLine 
    topLine = np.zeros((rightSideLength,2),dtype=np.int32)
    bottomLine = np.zeros((rightSideLength,2),dtype=np.int32)
    yTopbin=yCenterbin+1
    yBottombin=yCenterbin-1
    for i in range(0, rightSideLength, 1):
        topLine[i,0]=i+xCenterbin+1
        topLine[i,1]=yTopbin
        bottomLine[i,0]=i+xCenterbin+1
        bottomLine[i,1]=yBottombin
        if yTopbin < bins-1:
            yTopbin+=1
        if yBottombin > 0:
            yBottombin-=1
    bottomDens = 0
    for i in range(0,rightSideLength-1, 1):
        bottomDens+=smoothedHeatmap[bottomLine[i,0],[bottomLine[i,1]]]
    bottomDens=bottomDens/rightSideLength
    return bottomDens, leftLine, topLine, bottomLine

def Stat_GetMeanAndVariance_double(np.ndarray[double, ndim=1, mode="c"] input not None):
    cdef int m; #nSize
    m = input.shape[0];
    cdef double mean;
    cdef double var;
    mean=0.0
    var=0.0
    c_Stat_GetMeanAndVariance_double(&input[0], m, mean, var);
    return mean, var;

def findBin(heatmap, value, edges, scale='linear', T=1000):
    assert heatmap.shape[0] == heatmap.shape[1]
    nBins = heatmap.shape[0]
    vmin = min(edges)
    vmax = max(edges)
    if scale.lower()=='logish':
        value = convertToLogishPlotCoordinate(value, vmin, vmax, T)
    binIndex = (value-vmin)/(vmax-vmin) * nBins
    binIndex = int(round(binIndex,0))
    if binIndex<0:
        binIndex=0
    if binIndex>=nBins:
        binIndex=nBins-1
    return binIndex


#-----Some simple sample code to look at below------------

def test_multiply():
    print("testing!")

    a = np.arange(12, dtype=np.float64).reshape((3,4))

    print(a)

    multiply(a, 3)

    print(a)
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
def multiply(np.ndarray[double, ndim=2, mode="c"] input not None, double value):
    """
    multiply (arr, value)

    Takes a numpy arry as input, and multiplies each elemetn by value, in place

    param: array -- a 2-d numpy array of np.float64
    param: value -- a number that will be multiplied by each element in the array

    """
    cdef int m, n

    m, n = input.shape[0], input.shape[1]

    c_multiply(&input[0,0], value, m, n)

    return None

