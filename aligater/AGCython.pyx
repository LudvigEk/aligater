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

#for faster sorting with list comprehension
from operator import itemgetter


#AliGater imports
import aligater.AGConfig as agconf
from aligater.AGFileSystem import getGatedVector, getGatedVectors, reportGateResults, invalidAGgateParentError, invalidSampleError, filePlotError, AliGaterError, markerError
from aligater.AGPlotRoutines import addLine, addAxLine, getHeatmap, logishTransform, inverseLogishTransform, plotHeatmap, plot_densityFunc, convertToLogishPlotCoordinate, LogishLocator, LogishFormatter, BiLogLocator, BiLogFormatter, transformWrapper
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
def gateEllipsoid(fcsDF, str xCol, str yCol, float xCenter, float yCenter, list majorAxis, float majorRadii, list minorAxis, float minorRadii, vI=sentinel, str population="inner", bool info=True, scale='linear', T=1000):   
    #*************************************************Internal****************************************
    #******TODO: Should have wrapper, suggest width, height and angle in degrees with plotting********
    """
    Function that gates an ellipsoid given by direction and length of its axis'.
    Currently internal function, will produce a user friendly(ier) wrapper.
    
    Warning, if using a non-linear scale, the input center, axis and radii must be in the correct transformed coordinates.
    
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
    if xCol.lower()==yCol.lower():
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
    if majorRadii <= 0 or minorRadii <= 0:
        sys.stderr.write("in gateEllipsoid: Either minor or major Radii specified as zero/less than zero. Returning empty index.")
        return []
        
    vOutput=[]

    cdef np.ndarray[dtype_t, ndim = 1] vX = getGatedVector(fcsDF, xCol, vI, return_type='nparray',dtype=np.float64)
    cdef np.ndarray[dtype_t, ndim = 1] vY = getGatedVector(fcsDF, yCol, vI, return_type='nparray', dtype=np.float64)
    
    if scale.lower() != 'linear':    
        vX = transformWrapper(vX, T, scale)
        vY = transformWrapper(vY, T, scale)
    
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
        sys.stderr.write("WARNING, No events inside ellipsoid\n")
    if (len(vOutput) == 0 and population.lower() == "outer"):
        sys.stderr.write("WARNING, No events outside ellipsoid\n")
    if info:
        reportGateResults(vI, vOutput)
    return vOutput

@cython.wraparound(False)
@cython.boundscheck(False)
def gateThreshold(fcs, str name, str xCol, yCol=None, thresh=None, orientation="vertical", parentGate=None, population="upper", scale='linear', T=1000, filePlot=None, QC=False, *args, **kwargs):
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
    
    return outputGate


@cython.boundscheck(False)
@cython.wraparound(False)
def shortestPathMatrix(fcs, str name, str xCol, str yCol, list xboundaries, list yboundaries, parentGate=None, float sigma=3, int maxStep=20, str scale='linear', str xscale='linear', str yscale='linear', str startingCorner='bottomleft', str population='lower', int bins=300, float T=1000, QC=False, filePlot=None):
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    cdef list vI, originalvI
    if not fcs.__class__.__name__=="AGsample":
        raise TypeError("invalid AGsample object")
    if parentGate is None:
        vI=fcs.full_index()
        startingGate=None
    elif not parentGate.__class__.__name__ == "AGgate":
        raise TypeError("Parent population in "+name+" is an invalid AGgate object.")
    else:
        vI=parentGate()
        startingGate=parentGate
    if len(vI)<5:
        sys.stderr.write("Passed parent population to "+name+" contains too few events, returning empty gate.\n") 
        outputGate=AGgate([],parentGate,xCol,yCol,name)
        
        return outputGate
    
    cdef bool has_xbound, has_ybound
    has_xbound = has_ybound = False
    
    if len(xboundaries) ==0:
        pass
    elif len(xboundaries) !=2:
        reportStr="in shortestPathMatrix: xboundaries must be list of two, found: "+str(len(xboundaries))+" entries."
        raise ValueError(reportStr)
    else:
        has_xbound=True
        
    if len(yboundaries) ==0:
        pass
    elif len(yboundaries) !=2:
        reportStr="in shortestPathMatrix: yboundaries must be list of two, found: "+str(len(yboundaries))+" entries."
        raise ValueError(reportStr)
    else:
        has_ybound=True
        
    originalvI=vI
    fcsDF=fcs()
    if has_ybound and not has_xbound:
        tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=yboundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=startingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=yboundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=tmpvI, info=False)
    if has_xbound and not has_ybound:
        tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=xboundaries[0], orientation='vertical', population='upper',scale=scale, parentGate=startingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=xboundaries[1], orientation='vertical',population='lower',scale=scale,parentGate=tmpvI, info=False)
    if has_xbound and has_ybound:
        xtmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=xboundaries[0], orientation='vertical', population='upper',scale=scale, parentGate=startingGate,info=False)
        xstartingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=xboundaries[1], orientation='vertical',population='lower',scale=scale,parentGate=xtmpvI, info=False)
        ytmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=yboundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=xstartingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=yboundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=ytmpvI, info=False)
    #Switch from AGgate obj -> list from here
    vI=startingGate()
    cdef np.ndarray vX=getGatedVector(fcsDF, xCol, vI, return_type="nparray")
    cdef np.ndarray vY=getGatedVector(fcsDF, yCol, vI, return_type="nparray")
    
    if len(vX)<5 or len(vY)<5:
        sys.stderr.write("in shortestPathMatrix: For population "+name+". Too few events in specified window, returning empty gate.\n") 
        outputGate=AGgate([],parentGate,xCol,yCol,name)       
        return outputGate
    xscale = yscale = scale
    heatmap, xedges, yedges = getHeatmap(vX, vY, bins, scale, xscale, yscale, T)
    cdef np.ndarray[dtype_t, ndim=2] smoothedHeatmap = gaussian_filter(heatmap.astype(float),sigma=sigma)
    cdef np.ndarray[dtype_t, ndim=2] cost = np.empty_like(smoothedHeatmap, dtype=np.float_)
    cdef np.ndarray[dtype_t, ndim=2] leftBin = np.empty_like(smoothedHeatmap, dtype=np.float_)
    cost.fill(np.inf)
    leftBin.fill(0)




    
    cdef int y, x
    cdef int previousX, adjustedMaxStep, dist, stepIndex
    cdef float stepCost, penalty
    cdef int leftBinIndex
    cdef np.ndarray[dtype_t, ndim=1] stepCosts = np.zeros([maxStep],dtype=np.float_)
    cdef int costIdx, tmpIdx, minCostIdx
    cdef float minCost
    cdef int startY, endY
    #Force going from bottom left to top right
    if startingCorner.lower() == 'bottomleft':
        startX=0
        endX=bins
        xStep=1
        cost[0][0]=smoothedHeatmap[0][0]
        cost[bins-1][bins-1]=smoothedHeatmap[bins-1][bins-1]
    elif startingCorner.lower() == 'bottomright':
        startX=bins-1
        endX=-1
        xStep=-1
        maxStep=-maxStep
        cost[0][bins-1]=smoothedHeatmap[0][bins-1]
        cost[bins-1][0]=smoothedHeatmap[bins-1][0]

    for y in range(1,bins,1):
        for x in range(startX,endX,xStep): 
            for tmpIdx in range(0,abs(maxStep),1):
                stepCosts[tmpIdx]=10000000000000000.0
            if x-maxStep<0:
                adjustedMaxStep=x
            elif x-maxStep>bins-1:
                adjustedMaxStep=-(bins-1-x)
            else:
                adjustedMaxStep=maxStep-xStep
            costIdx=0

            for previousX in range(x-adjustedMaxStep,x+xStep,xStep):
                dist=x-previousX
                penalty=dist*dist
                stepCost = cost[y-1,previousX]+smoothedHeatmap[y,x]+penalty
                stepCosts[costIdx]=stepCost
                costIdx+=1

            minCost=10000000000000000.0

            for tmpIdx in range(0,abs(maxStep),1):
                if stepCosts[tmpIdx]<minCost:
                    minCost=stepCosts[tmpIdx]
                    minCostIdx=tmpIdx*xStep #addition
                    
            cost[y,x]=minCost
            leftBinIndex=minCostIdx+x-adjustedMaxStep #why subtract adjustedMaxstep here??
            leftBin[y,x]=leftBinIndex
            
    if startingCorner.lower() == 'bottomright':
        leftBinIndex=0
    #traverse cost matrix, /w monothony
    if leftBinIndex < 0:
        leftBinIndex = 0
    if leftBinIndex > bins-1:
        leftBinIndex = bins-1
    
    cdef list path=[]
    for x in range(bins-1,-1,-1):
        path.append([xedges[x],yedges[leftBinIndex]])
        leftBinIndex=int(leftBin[x][leftBinIndex])
    
    path=path[::-1]   

    cdef int count=0
    cdef list coord
    if plot or filePlot is not None:
        heatmap=np.ma.masked_where(smoothedHeatmap == 0, smoothedHeatmap)
        plt.clf()
        fig, ax = plt.subplots()
        extent = [xedges[0], xedges[bins], yedges[0], yedges[bins]]
        plt.imshow(heatmap.T, extent=extent, origin='lower',aspect='auto')
        plt.xlabel(xCol)
        plt.ylabel(yCol)
        cmap=plt.get_cmap()
        cmap.set_bad(color='white')
        if scale.lower()=='logish':
            ax=plt.gca()
            ax.xaxis.set_major_locator(LogishLocator(linCutOff=T))
            ax.xaxis.set_major_formatter(LogishFormatter(linCutOff=T))
            ax.yaxis.set_major_locator(LogishLocator(linCutOff=T))
            ax.yaxis.set_major_formatter(LogishFormatter(linCutOff=T))
        elif scale.lower()=='bilog':
            ax=plt.gca()
            ax.xaxis.set_major_locator(BiLogLocator(linCutOff=T))
            ax.xaxis.set_major_formatter(BiLogFormatter(linCutOff=T))
            ax.yaxis.set_major_locator(BiLogLocator(linCutOff=T))
            ax.yaxis.set_major_formatter(BiLogFormatter(linCutOff=T))
        else:
            pass
             
        #draw line of shortest path
        for coord in path:
            if count==0:
                previousCoord=coord
                count+=1
                continue 
            fig,ax = addLine(fig,ax,previousCoord,coord,scale=scale,T=T)
            previousCoord=coord
            count+=1
        if filePlot is not None:
            plt.savefig(filePlot)
        if plot:
            plt.show()
        plt.close(fig)
    #print(path)
    vOut=gatePointList(fcsDF,xCol,yCol,path, population=population, vI=originalvI)
    reportGateResults(originalvI,vOut)
    
    if plot:
        plt.clf()
        plotHeatmap(fcsDF, xCol, yCol, vOut, scale=scale)
        plt.show()
    if parentGate is not None:
        outputGate=AGgate(vOut, parentGate, xCol, yCol, name)
    else:
        outputGate=AGgate(vOut, None, xCol, yCol, name)
        
    return outputGate



# def ___OLD___horisontalPath(fcs, str name, str xCol, str yCol, parentGate=None, population='negative',
#                  startY=None, list xboundaries=None, list yboundaries=None, bool leftRight=True , str direction='up', maxStep=5, phi=0,
#                  int bins=300, float sigma=3, str scale='linear', int T=1000, bool plot=True):
#     if agconf.execMode in ["jupyter","ipython"]:
#         plot=True
#     else:
#         plot=False
#     cdef list vI, originalvI
#     if not fcs.__class__.__name__=="AGsample":
#         raise TypeError("invalid AGsample object")
#     if parentGate is None:
#         vI=fcs.full_index()
#         startingGate=None
#     elif not parentGate.__class__.__name__ == "AGgate":
#         raise TypeError("Parent population in "+name+" is an invalid AGgate object.")
#     else:
#         vI=parentGate()
#         startingGate=parentGate
#     if len(vI)<5:
#         sys.stderr.write("Passed parent population to "+name+" contains too few events, returning empty gate.\n") 
#         outputGate=AGgate([],parentGate,xCol,yCol,name)
#     cdef int startbin
#     cdef bool has_xbound, has_ybound
#     has_xbound = has_ybound = False
    
#     if xboundaries is not None:
#         if isinstance(xboundaries, list):
#             if len(xboundaries) ==0:
#                 pass
#             elif len(xboundaries) !=2:
#                 reportStr="in horisontalPath: xboundaries must be list of two, found: "+str(len(xboundaries))+" entries."
#                 raise ValueError(reportStr)
#             else:
#                 has_xbound=True
#         else:
#             raise AliGaterError("in horisontalPath: ","is xboundaries is passed it must be list.")

#     if yboundaries is not None:
#         if isinstance(yboundaries, list):
#             if len(yboundaries) ==0:
#                 pass
#             elif len(yboundaries) !=2:
#                 reportStr="in horisontalPath: yboundaries must be list of two, found: "+str(len(yboundaries))+" entries."
#                 raise ValueError(reportStr)
#             else:
#                 has_ybound=True    
#         else:
#             raise AliGaterError("in horisontalPath: ","is yBoundaries is passed it must be list.")
    
#     originalvI=vI
#     if has_ybound and not has_xbound:
#         tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=yboundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=startingGate,info=False)
#         startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=yboundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=tmpvI, info=False)
#     if has_xbound and not has_ybound:
#         tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=xboundaries[0], orientation='vertical', population='upper',scale=scale, parentGate=startingGate,info=False)
#         startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=xboundaries[1], orientation='vertical',population='lower',scale=scale,parentGate=tmpvI, info=False)
#     if has_xbound and has_ybound:
#         xtmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=xboundaries[0], orientation='vertical', population='upper',scale=scale, parentGate=startingGate,info=False)
#         xstartingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=xboundaries[1], orientation='vertical',population='lower',scale=scale,parentGate=xtmpvI, info=False)
#         ytmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=yboundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=xstartingGate,info=False)
#         startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=yboundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=ytmpvI, info=False)
#     #Switch from AGgate obj -> list from here
#     vI=startingGate()

#     for y in range(1,bins,1):
#         for x in range(startX,endX,xStep): 
#             for tmpIdx in range(0,abs(maxStep),1):
#                 stepCosts[tmpIdx]=10000000000000000.0
#             if x-maxStep<0:
#                 adjustedMaxStep=x
#             elif x-maxStep>bins-1:
#                 adjustedMaxStep=-(bins-1-x)
#             else:
#                 adjustedMaxStep=maxStep-xStep
#             costIdx=0

#             for previousX in range(x-adjustedMaxStep,x+xStep,xStep):
#                 dist=x-previousX
#                 penalty=dist*dist
#                 stepCost = cost[y-1,previousX]+smoothedHeatmap[y,x]+penalty
#                 stepCosts[costIdx]=stepCost
#                 costIdx+=1

#             minCost=10000000000000000.0

#             for tmpIdx in range(0,abs(maxStep),1):
#                 if stepCosts[tmpIdx]<minCost:
#                     minCost=stepCosts[tmpIdx]
#                     minCostIdx=tmpIdx*xStep #addition
                    
#             cost[y,x]=minCost
#             leftBinIndex=minCostIdx+x-adjustedMaxStep #why subtract adjustedMaxstep here??
#             leftBin[y,x]=leftBinIndex
            
#     if startingCorner.lower() == 'bottomright':
#         leftBinIndex=0
#     #traverse cost matrix, /w monothony
#     if leftBinIndex < 0:
#         leftBinIndex = 0
#     if leftBinIndex > bins-1:
#         leftBinIndex = bins-1
    
#     #leftBinIndex = bins-1 - maxStep
#     cdef list path=[]
#     for x in range(bins-1,-1,-1):
#         path.append([xedges[x],yedges[leftBinIndex]])
#         leftBinIndex=int(leftBin[x][leftBinIndex])
    
#     path=path[::-1]   

#     print(leftBin)
#     cdef int count=0
#     cdef list coord
#     if plot:
#         heatmap=np.ma.masked_where(smoothedHeatmap == 0, smoothedHeatmap)
#         plt.clf()
#         fig, ax = plt.subplots()
#         extent = [xedges[0], xedges[bins], yedges[0], yedges[bins]]
#         plt.imshow(heatmap.T, extent=extent, origin='lower',aspect='auto')
#         plt.xlabel(xCol)
#         plt.ylabel(yCol)
#         cmap=plt.get_cmap()
#         cmap.set_bad(color='white')
#         if scale.lower()=='logish':
#             ax=plt.gca()
#             ax.xaxis.set_major_locator(LogishLocator(linCutOff=T))
#             ax.xaxis.set_major_formatter(LogishFormatter(linCutOff=T))
#             ax.yaxis.set_major_locator(LogishLocator(linCutOff=T))
#             ax.yaxis.set_major_formatter(LogishFormatter(linCutOff=T))
#         elif scale.lower()=='bilog':
#             ax=plt.gca()
#             ax.xaxis.set_major_locator(BiLogLocator(linCutOff=T))
#             ax.xaxis.set_major_formatter(BiLogFormatter(linCutOff=T))
#             ax.yaxis.set_major_locator(BiLogLocator(linCutOff=T))
#             ax.yaxis.set_major_formatter(BiLogFormatter(linCutOff=T))
        
             
#         #draw line of shortest path
#         for coord in path:
#             if count==0:
#                 previousCoord=coord
#                 count+=1
#                 continue 
#             fig,ax = addLine(fig,ax,previousCoord,coord,scale=scale,T=T)
#             previousCoord=coord
#             count+=1
#         plt.show()
#     #print(path)
#     vOut=gatePointList(fcsDF,xCol,yCol,path, population=population, vI=originalvI)
#     reportGateResults(originalvI,vOut)
    
#     if plot:
#         plt.clf()
#         plotHeatmap(fcsDF, xCol, yCol, vOut, scale=scale)
#         plt.show()
#     if parentGate is not None:
#         outputGate=AGgate(vOut, parentGate, xCol, yCol, name)
#     else:
#         outputGate=AGgate(vOut, fcs.full_index(), xCol, yCol, name)
        
#     return outputGate

def horisontalPath(fcs, str name, str xCol, str yCol, parentGate=None, population='negative',
                 startY=None, endY=None, list xboundaries=None, list yboundaries=None, bool leftRight=True , str direction='up', maxStep=5, phi=0,
                 int bins=300, float sigma=3, str scale='linear', int T=1000, bool plot=True):
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
        startingGate=parentGate
    if len(vI)<5:
        sys.stderr.write("Passed parent population to "+name+" contains too few events, returning empty gate.\n") 
        outputGate=AGgate([],parentGate,xCol,yCol,name)
    #cdef int startbin
    #originalvI=vI

#*********************************************************************************************************
    cdef int startbin
 
    cdef bool has_xbound, has_ybound
    has_xbound = has_ybound = False
    
    if xboundaries is not None:
        if isinstance(xboundaries, list):
            if len(xboundaries) ==0:
                pass
            elif len(xboundaries) !=2:
                reportStr="in horisontalPath: xboundaries must be list of two, found: "+str(len(xboundaries))+" entries."
                raise ValueError(reportStr)
            else:
                has_xbound=True
        else:
            raise AliGaterError("in horisontalPath: ","is xboundaries is passed it must be list.")

    if yboundaries is not None:
        if isinstance(yboundaries, list):
            if len(yboundaries) ==0:
                pass
            elif len(yboundaries) !=2:
                reportStr="in horisontalPath: yboundaries must be list of two, found: "+str(len(yboundaries))+" entries."
                raise ValueError(reportStr)
            else:
                has_ybound=True    
        else:
            raise AliGaterError("in horisontalPath: ","is yBoundaries is passed it must be list.")
    
    originalvI=vI
    if has_ybound and not has_xbound:
        tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=yboundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=startingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=yboundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=tmpvI, info=False)
    if has_xbound and not has_ybound:
        tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=xboundaries[0], orientation='vertical', population='upper',scale=scale, parentGate=startingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=xboundaries[1], orientation='vertical',population='lower',scale=scale,parentGate=tmpvI, info=False)
    if has_xbound and has_ybound:
        xtmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=xboundaries[0], orientation='vertical', population='upper',scale=scale, parentGate=startingGate,info=False)
        xstartingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=xboundaries[1], orientation='vertical',population='lower',scale=scale,parentGate=xtmpvI, info=False)
        ytmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=yboundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=xstartingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=yboundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=ytmpvI, info=False)
    #Switch from AGgate obj -> list from here
    if startingGate is not None:
        vI=startingGate()
    else:
        vI=fcs.full_index()
#*********************************************************************************************************



    vX,vY = getGatedVectors(fcsDF=fcs(), gate1=xCol, gate2=yCol, vI=vI)
    #Note on the heatmap, from numpy docs, np.histogram2d
    #Please note that the histogram does not follow the Cartesian convention where x values are on the abscissa and y values on the ordinate axis. 
    #Rather, x is histogrammed along the first dimension of the array (vertical), and y along the second dimension of the array (horisontal).
    heatmap, xedges, yedges = getHeatmap(vX=vX, vY=vY, scale=scale, normalize=True, xscale=scale, yscale=scale, T=T, bins=bins)
    #I.e. note here, in contrast to verticalPath the cost heatmap from np.histogram2d is NOT transposed!
    cdef np.ndarray[dtype_t, ndim=2] cost = gaussian_filter(heatmap.astype(float),sigma=sigma)
    
    cdef np.ndarray[dtype_t, ndim=2] plot_heatmap = np.copy(cost)
        
    #Adjust phi based on avg val in cost
    cdef float mean_heatmap_val
    mean_heatmap_val = np.mean(cost)
    cdef corrected_phi = phi * mean_heatmap_val
    
    if direction.lower() == 'up':
        minCostPathDir = 'right'
    elif direction.lower() == 'down':
        minCostPathDir = 'left'
    elif direction.lower() == 'both':
        minCostPathDir = direction
    else: 
        raise

    for i in np.arange(0, len(yedges)-2, 1):
        if startY >= yedges[i] and startY < yedges[i+1]:
            startBin = i
            break
    else:
        raise AliGaterError("in horisontalPath","startY out of bounds")

    cdef int endBin
    if endY is None:
        endY=startY
    for i in np.arange(0, len(yedges)-2, 1):
        if endY >= yedges[i] and endY < yedges[i+1]:
            endBin = i
            break
    else:
        raise AliGaterError("in horisontalPath","startY out of bounds")

    cdef float LARGE_NUMBER = 10000000000000000.0
    
#    if not leftRight:
#        #To be able go right-left and not just left-right there's an option to reverse the rows order and keep internal functions
#        reverse = True
#        for i in np.arange(0, len(cost[-1]),1):
#            if not i == endBin:
#                cost[-1][i]=LARGE_NUMBER
#            else:
#                cost[-1][i]=0        
#    else:
#        for i in np.arange(0, len(cost[0]),1):
#            if not i == endBin:
#                cost[0][i]=LARGE_NUMBER
#            else:
#                cost[0][i]=0
#        reverse=False

    if not leftRight:
        #To be able go right-left and not just left-right there's an option to reverse the rows order and keep internal functions
        reverse = False
        for i in np.arange(0, len(cost[0]),1):
            if not i == endBin:
                cost[0][i]=LARGE_NUMBER
            else:
                cost[0][i]=0        
    else:
        for i in np.arange(0, len(cost[-1]),1):
            if not i == endBin:
                cost[-1][i]=LARGE_NUMBER
            else:
                cost[-1][i]=0
        reverse=True

    
    pathMatrix = _minCostPath(cost, nCols=bins, nRows=bins, maxStep=maxStep, reverse=reverse, direction=minCostPathDir, phi=corrected_phi)
    
    
    cdef int curBin
    cdef int nextBin
    path=[]
    curBin=startBin
    row_order = np.arange(0, bins, 1)
    #if not leftRight: #ORIGINAL
    if not leftRight:
        row_order = row_order[::-1]
    for rowIdx in row_order:
        nextBin = pathMatrix[rowIdx][curBin]
        path.append( [ xedges[rowIdx], yedges[curBin]] )
        curBin=nextBin
    if leftRight:
        path.append( [xedges[-1], yedges[curBin]])
    else:
        path.append( [xedges[0], yedges[curBin]])
        
    #if not leftRight:
    #    path = path[::-1]
        
    vOut=gatePointList(fcsDF=fcs(),xCol=xCol,yCol=yCol,vPL=path, population=population, bHorisontal=True, vI=originalvI)
    reportGateResults(vI,vOut)
    

    if parentGate is not None:
        outputGate=AGgate(vOut, parentGate, xCol, yCol, name)
    else:
        outputGate=AGgate(vOut, None, xCol, yCol, name)
        
    if plot:
        plt.clf()
        fig, ax = plt.subplots()
        extent = [min(xedges), max(xedges), min(yedges), max(yedges)]
        plt.imshow(plot_heatmap.T, extent=extent, origin='lower',aspect='auto')
        plt.xlabel(xCol)
        plt.ylabel(yCol)
        cmap=plt.get_cmap()
        cmap.set_bad(color='white')
        count=0
        for coord in path:
            if count==0:
                previousCoord=coord
                count+=1
                continue 
            fig,ax = addLine(fig,ax,previousCoord,coord,scale=scale,T=T)
            previousCoord=coord
            count+=1
        plt.show()
        plt.clf()
        plotHeatmap(fcs(), xCol, yCol, outputGate(), scale=scale,thresh=T)
        plt.show()
        
    return outputGate

def verticalPath(fcs, str name, str xCol, str yCol, parentGate=None, population='lower',
                 startX=None, endX=None, bool bottomUp=True, list xboundaries=None, list yboundaries=None, str direction='right', bool openEnded=False, maxStep=5, phi=0.0,
                 int bins=300, float sigma=3, str scale='linear', int T=1000, bool plot=True, filePlot=None):
    if agconf.execMode in ["jupyter","ipython"]:
        plot=True
    else:
        plot=False
    cdef list vI, originalvI
    if not fcs.__class__.__name__=="AGsample":
        raise TypeError("invalid AGsample object")
    #TODO HANDLE START/END X TYPES
    if parentGate is None:
        vI=fcs.full_index()
        startingGate = None
    elif not parentGate.__class__.__name__ == "AGgate":
        raise TypeError("Parent population in "+name+" is an invalid AGgate object.")
    else:
        vI=parentGate()
        startingGate=parentGate
    if len(vI)<5:
        sys.stderr.write("Passed parent population to "+name+" contains too few events, returning empty gate.\n") 
        outputGate=AGgate([],parentGate,xCol,yCol,name)
        return outputGate

    cdef int startbin
 
    cdef bool has_xbound, has_ybound
    has_xbound = has_ybound = False
    
    if xboundaries is not None:
        if isinstance(xboundaries, list):
            if len(xboundaries) ==0:
                pass
            elif len(xboundaries) !=2:
                reportStr="in verticalPath: xboundaries must be list of two, found: "+str(len(xboundaries))+" entries."
                raise ValueError(reportStr)
            else:
                has_xbound=True
        else:
            raise AliGaterError("in verticalPath: ","is xboundaries is passed it must be list.")

    if yboundaries is not None:
        if isinstance(yboundaries, list):
            if len(yboundaries) ==0:
                pass
            elif len(yboundaries) !=2:
                reportStr="in verticalPath: yboundaries must be list of two, found: "+str(len(yboundaries))+" entries."
                raise ValueError(reportStr)
            else:
                has_ybound=True    
        else:
            raise AliGaterError("in verticalPath: ","is yBoundaries is passed it must be list.")
    
    originalvI=vI
    if has_ybound and not has_xbound:
        tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=yboundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=startingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=yboundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=tmpvI, info=False)
    if has_xbound and not has_ybound:
        tmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=xboundaries[0], orientation='vertical', population='upper',scale=scale, parentGate=startingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=xboundaries[1], orientation='vertical',population='lower',scale=scale,parentGate=tmpvI, info=False)
    if has_xbound and has_ybound:
        xtmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=xboundaries[0], orientation='vertical', population='upper',scale=scale, parentGate=startingGate,info=False)
        xstartingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=xboundaries[1], orientation='vertical',population='lower',scale=scale,parentGate=xtmpvI, info=False)
        ytmpvI=gateThreshold(fcs, name="tmpvI", xCol=xCol, yCol=yCol, thresh=yboundaries[0], orientation='horisontal', population='upper',scale=scale, parentGate=xstartingGate,info=False)
        startingGate=gateThreshold(fcs,name="vI",xCol=xCol,yCol=yCol, thresh=yboundaries[1], orientation='horisontal',population='lower',scale=scale,parentGate=ytmpvI, info=False)
    #Switch from AGgate obj -> list from here
    if startingGate is not None:
        vI=startingGate()
    else:
        vI=fcs.full_index()
    
    vX,vY = getGatedVectors(fcsDF=fcs(), gate1=xCol, gate2=yCol, vI=vI)
    
    heatmap, xedges, yedges = getHeatmap(vX=vX, vY=vY, scale=scale, normalize=True, xscale=scale, yscale=scale, T=T, bins=bins)
    #Note on heatmap, from numpy docs, np.histogram2d
    #Please note that the histogram does not follow the Cartesian convention where x values are on the abscissa and y values on the ordinate axis. 
    #Rather, x is histogrammed along the first dimension of the array (vertical), and y along the second dimension of the array (horizontal). 
    #Therefore, we transpose to get x on cols and y on rows
    cdef np.ndarray[dtype_t, ndim=2] cost = gaussian_filter(heatmap.astype(float),sigma=sigma).T
    cdef np.ndarray[dtype_t, ndim=2] plot_heatmap = np.copy(cost)
    
    #Adjust phi based on avg val in cost
    cdef float mean_heatmap_val
    mean_heatmap_val = np.mean(cost)
    cdef corrected_phi = phi * mean_heatmap_val


    #find start/endbin
    for i in np.arange(0, len(xedges)-2, 1):
        if startX >= xedges[i] and startX < xedges[i+1]:
            startBin = i
            break
    else:
        raise AliGaterError("in verticalPath","startX out of bounds")    

    cdef int endBin
    if endX is None:
        endBin = startBin
    else:
        for i in np.arange(0, len(xedges)-2, 1):
            if endX >= xedges[i] and endX < xedges[i+1]:
                endBin = i
                break
        else:
            raise AliGaterError("in verticalPath","endX out of bounds")
            
    cdef float LARGE_NUMBER = 10000000000000000.0
    
                
    if bottomUp:        
        reverse=True
        for i in np.arange(0, len(cost[-1]),1):
            if not i == endBin and not openEnded:
                cost[-1][i]=LARGE_NUMBER
            else:
                cost[-1][i]=0
    else:
        for i in np.arange(0, len(cost[0]),1):
            if not i == endBin and not openEnded:
                cost[0][i]=LARGE_NUMBER
            else:
                cost[0][i]=0
        reverse=False
    
    pathMatrix = _minCostPath(cost, nCols=bins, nRows=bins, maxStep=maxStep, reverse=reverse, direction=direction, phi=corrected_phi)
    
    
    cdef int curBin
    cdef int nextBin
    path=[]
    curBin=startBin

    row_order = np.arange(0, bins, 1)
    if not bottomUp: 
        row_order = row_order[::-1]
    for rowIdx in row_order:
        nextBin = pathMatrix[rowIdx][curBin]
        path.append( [ xedges[curBin], yedges[rowIdx]] )
        curBin=nextBin
    if bottomUp: 
        path.append( [xedges[curBin], yedges[-1]])
    else:
        path.append( [xedges[curBin], yedges[0]])
    
    vOut=gatePointList(fcs(),xCol,yCol,path, population=population, vI=originalvI, bHorisontal=False)
    reportGateResults(vI,vOut)
    

    if parentGate is not None:
        outputGate=AGgate(vOut, parentGate, xCol, yCol, name)
    else:
        outputGate=AGgate(vOut, None, xCol, yCol, name)
        
    if plot or filePlot is not None:
        plt.clf()
        fig, ax = plt.subplots()
        extent = [min(xedges), max(xedges), min(yedges), max(yedges)]
        plt.imshow(plot_heatmap, extent=extent, origin='lower',aspect='auto')
        plt.xlabel(xCol)
        plt.ylabel(yCol)
        cmap=plt.get_cmap()
        cmap.set_bad(color='white')
        count=0
        for coord in path:
            if count==0:
                previousCoord=coord
                count+=1
                continue 
            fig,ax = addLine(fig,ax,previousCoord,coord,scale=scale,T=T)
            previousCoord=coord
            count+=1
        if filePlot is not None:
            plt.savefig(filePlot)
            
        if plot:
            plt.show()
            
        plt.clf()
        
        if plot:
            plotHeatmap(fcs(), xCol, yCol, outputGate(), scale=scale,thresh=T)
            plt.show()
        
    return outputGate

def _minCostPath(np.ndarray[dtype_t, ndim=2] cost, int nCols, int nRows, int maxStep=5, bool reverse = False, float phi=0.0, str direction='right') :
#A type of gradient descent to find the best next step from every cell of a matrix, given a direction
#Requirements on monothony and always progress (i.e only one + maxStep possible steps from any cell, straight down and up to maxStep sideways)
#pathMatrix is a matrix with the same dimensions as cost, with each cell containing the index of the element in the next row which is the cheapest to jump to
#Todo: Bellmans optimality principle(?)
    
    pathMatrix = np.empty_like(cost, dtype=np.float64)
    dirArr = np.empty_like(cost, dtype=np.float32)
    pathMatrix[0][0] = 0#cost[0][0]; 
  
    # Construct tc array 
    cdef list row_order 
    if reverse:
        #row_order = row_order[::-1]
        row_order = list(range(nRows-1, -1, -1))
        rowStep=-1
    else:
        row_order = list(range(0, nRows, 1))
        rowStep=1

    cdef float score=0
    for row_idx in row_order:         #(i = 1; i <= m; i++):
        for col_idx in range(0, nCols, 1):      #(j = 1; j <= n; j++):
            if row_idx != nRows-1 and row_idx != 0:
                #If not last row
                #score, pathMatrix[row_idx][col_idx] = _minCostPath_nextStep(curRow=cost[row_idx], nextRow=cost[row_idx+rowStep], curPos=col_idx, maxStep=maxStep, phi=phi, direction=direction)       #old
                cost[row_idx][col_idx], pathMatrix[row_idx][col_idx] = _minCostPath_nextStep(curRow=cost[row_idx], nextRow=cost[row_idx-rowStep], curPos=col_idx, maxStep=maxStep, phi=phi, direction=direction)       #[1]
            else:
                pathMatrix[row_idx][col_idx] = col_idx


    return pathMatrix
 
def _minCostPath_nextStep(curRow, nextRow, int curPos, int maxStep, float phi=0, str direction='right'):
    #TODO: Possibly introduce way to estimate phi by looking at medium density of hisotgram density
    
    #Takes a current row in a cost matrix and the next row, and calculates the cheapest step possible
    #Accepts direction left or right relative to the current position, but always take one step down in the cost matrix (y-axis)
    #Returns the position of the best step in the next row
    #
    #i.e you are at X
    #********X***********
    #********|-----|*****
    #Theres a range of allowed steps
    #Find the best and return the position in that row
    #Theres a penalty for taking larger steps, 
    #But that can be accepted if the cost at that position is much lower
    #Because of the monotonous requirement, a step may never go past the end point (m)    
    
    if direction.lower() not in ['right', 'left', 'both']:
        raise AliGaterError("in minCostPath_nextStep", "direction must be either of 'right', 'left' or 'both'. Default: 'right'")
        
    #if m < 0 or m >= len(nextRow):
    #     print(m)
    #     print(len(nextRow))
    #     raise AliGaterError("in minCostPath_nextStep", "specified end position (m) is out of bounds")
    #Boundscheck for maxStep, is it even allowed to take a max step?
    #NOTE: maxStep is changed from being 'the maximum allowed distance' to...
    #the index of the maximum allowed step on the next row (makes easier to read for loops later)
    maxStepLeft = None
    maxStepRight = None
    
    if (curPos + maxStep) >= (len(curRow)-1) and direction.lower() == 'right':
        maxStep = curPos + (len(curRow) - 1 - curPos)
    elif direction.lower() == 'right':
        maxStep = curPos + maxStep
  
    if (curPos - maxStep) < 0 and direction.lower() == 'left':
        maxStep = 0
    elif direction.lower() == 'left':
        maxStep = curPos - maxStep
    
    if direction.lower() == 'both':
        maxStepLeft = curPos - maxStep
        maxStepRight = curPos + maxStep
        if (maxStepRight) >= (len(curRow)-1):
            maxStepRight = curPos + (len(curRow) - 1 - curPos)
        if (maxStepLeft) < 0:
            maxStepLeft = 0     
        
    #We save the possible steps in a list of tuple (index of that step, and the cost)
    #Then sort that list by the 2nd var and pick the top (cheapest) step
    cdef list stepCosts = []
    cdef int bestStepIndex
    cdef float bestStepCost
    
    if direction.lower() == 'right':
        for i in range(curPos, maxStep+1, 1):
            stepCosts.append( (i, curRow[curPos] + nextRow[i] + phi*((i-curPos)**2)) ) 
        bestStepTuple = sorted(stepCosts, key=itemgetter(1))
        bestStepCost = bestStepTuple[0][1]
        bestStepIdx=bestStepTuple[0][0]
    elif direction.lower() == 'left':
        for i in range(curPos, maxStep-1, -1):
            stepCosts.append( (i,curRow[curPos] + nextRow[i] + phi*((curPos-i)**2)) ) 
        bestStepTuple = sorted(stepCosts, key=itemgetter(1))
        bestStepIdx=bestStepTuple[0][0]
        bestStepCost = bestStepTuple[0][1]
    else: #i.e direction = both        
        for i in range(maxStepLeft, maxStepRight+1, 1):
            stepCosts.append( (i, curRow[curPos] + nextRow[i] + phi*((curPos - i)**2)) )
        bestStepTuple = sorted(stepCosts, key=itemgetter(1))
        bestStepIdx=bestStepTuple[0][0]
        bestStepCost = bestStepTuple[0][1]
        
    return bestStepCost, bestStepIdx


def test_minCost(cost, m, n, nRrow, nCol): 
  
    # Instead of following line, we can use int tc[m+1][n+1] or 
    # dynamically allocate memoery to save space. The following 
    # line is used to keep te program simple and make it working 
    # on all compilers. 
    tc = [[0 for x in range(nCol)] for x in range(nRrow)] 
  
    tc[0][0] = cost[0][0] 
  
    # Initialize first column of total cost(tc) array 
    for i in range(1, m+1): 
        tc[i][0] = tc[i-1][0] + cost[i][0] 
  
    # Initialize first row of tc array 
    for j in range(1, n+1): 
        tc[0][j] = tc[0][j-1] + cost[0][j] 
  
    # Construct rest of the tc array 
    for i in range(1, m+1): 
        for j in range(1, n+1): 
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] 
  
    return tc[m][n] 




#@cython.boundscheck(False)
#@cython.wraparound(False)
def gatePointList(fcsDF, xCol, yCol, vPL, population='lower',vI=sentinel, bHorisontal=True, scale='linear', T=1000):
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
    if scale.lower() not in ['linear', 'logish', 'bilog']:
        raise("specify scale as either 'linear', 'logish' or 'bilog'\n")
    vOut=[]
    vX=getGatedVector(fcsDF, xCol, vI,return_type='nparray')
    vY=getGatedVector(fcsDF, yCol, vI,return_type='nparray')
    
    cdef int i
    if scale.lower() != 'linear':
        vX=transformWrapper(vX, scale=scale, T=T)
        vY=transformWrapper(vY, scale=scale, T=T)
        for i in np.arange(0, len(vPL), 1):
            vPL[i]=transformWrapper(vPL[i], scale=scale, T=T)
        
    assert len(vX) == len(vY) == len(vI)
    
    cdef float x0=min(vX)
    cdef float x1=max(vX)
    cdef int nOfBins=len(vPL)
    cdef float lim_coord, lim #lim, lim_coord will be x or y depending on axis of interest
    cdef int index
    cdef int dist
    cdef bool bUnhandled = False
    
    cdef int targetAxis = 1 #1 = y-sep, 0 = x-sep
    cdef int binAxis = 0 #0 = y-sep, 1 = x-sep
    
    if bHorisontal:
        targetAxis = 1
        binAxis = 0
    else:
        targetAxis = 0
        binAxis = 1

    #Create construct of [[x, y, index], [x,y,index], ...]
    events = np.dstack([vX, vY, vI])[0]
    #sort events by separating coordinate (by x if separated by y (horisontal line))
    sorted_events = sorted(events, key=itemgetter(binAxis))
    #Sort limit construct by non-separating coordinate (by y if separated by x (vertical line))
    sorted_vPL = sorted(vPL, key=itemgetter(binAxis))
    
    #Note on bUnhandled, if a give line does not span the full gating view... 
    #then before/after the first/last point in that line, extented its limits in the correct direction (horisontally for horisontal lines, vis a vis vertical)

    i=0
    cdef int idx_vPL=0
    while i<len(sorted_events):
        if idx_vPL == len(vPL):
            bUnhandled=True
            break
        lim = sorted_vPL[idx_vPL][targetAxis]
        bin_i = sorted_vPL[idx_vPL][binAxis]
        
        while sorted_events[i][binAxis] < bin_i:
            lim_coord=sorted_events[i][targetAxis]
            index=sorted_events[i][2]
            if lim_coord < lim:
                if lower:
                    vOut.append(index)
            if lim_coord >= lim:
                if not lower:
                    vOut.append(index)

            i+=1
            #Since inner while loop is based on the bin limits its possible for it to go out of bounds in some cases
            #Below if statement safeguards against that
            if i >= len(sorted_events):
                break
        idx_vPL+=1

    if bUnhandled:
        lim = sorted_vPL[len(vPL)-1][targetAxis]
        while i < len(sorted_events):
            lim_coord = sorted_events[i][targetAxis]
            index=sorted_events[i][2]
            if lim_coord < lim:
                if lower:
                    vOut.append(index)
            if lim_coord >= lim:
                if not lower:
                    vOut.append(index)
            i+=1
            
    return vOut        
    #OLD SOUTION BELOW
    # for event in np.arange(0,len(vX),1):
        
    #     x = vX[event]
    #     y = vY[event]
    #     index = vI[event]
    #     #dist = round( (x-x0)/(x1-x0) * nOfBins)
    #     #*********************rework to deal with non-linear bins, as is case with transformed scales***************************
    #     dist = np.searchsorted(xBinLims,x)
    #     # for i in np.arange(0,nOfBins-2,1):
    #     #     if x < vPL[0][0]:
    #     #         dist=0
    #     #         break
    #     #     elif x >= vPL[i][0] and x <= vPL[i+1][0]:
    #     #         dist=i
    #     #         break
    #     #     elif x > vPL[nOfBins-1][0]:
    #     #         dist=nOfBins-1
    #     #         break
    #     # else:
    #     #     raise
    #     #***********************************************************************************************************************
    #     if dist > nOfBins-1:
    #         dist=nOfBins-1
    #     ylim=vPL[dist][1]

    #     if y<ylim:
    #         if lower:
    #             vOut.append(index)
    #     if y>=ylim:
    #         if not lower:
    #             vOut.append(index)

    # return vOut

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

