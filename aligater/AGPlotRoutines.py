#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#			~~aliGater~~
#	(semi)automated gating software
#
#               /^^\
#   /^^\_______/0  \_
#  (                 `~+++,,_________,,++~^^^^^^^
# ..V^V^V^V^V^V^\.................................
#
#
#	Parsing .fcs files with fcsparser from Eugene Yurtsevs FlowCytometryTools (very slightly modified)
#	Check out his toolkit for flow cytometry analysis: 
#	http://eyurtsev.github.io/FlowCytometryTools/
#
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	https://www.med.lu.se/labmed/hematologi_och_transfusionsmedicin/forskning/bjoern_nilsson
#   Distributed under the MIT License

import pandas as pd
import numpy as np

import math
import six
from scipy.ndimage.filters import gaussian_filter1d
# For computing bin width similarly to scipys histogram_bin_edges
from scipy.stats import iqr
from sklearn.decomposition import PCA

import sys

# AliGater imports
import aligater.AGConfig as agconf
from aligater.AGFileSystem import getGatedVector, AliGaterError
from aligater.AGCythonUtils import __vectorlogicleTransform, __vectorInverselogicleTransform, __vectorBilogTransform, \
    __vectorInverseBilogTransform

import matplotlib

if agconf.execMode == "jupyter":
    # Use default backend/Let matplotlib decide
    pass
else:
    matplotlib.use(
        'Agg')  # Solves tkinter backend problems using ray multithreading - gives some problems in notebooks though

from matplotlib import pyplot as plt
import matplotlib.colors
from matplotlib.patches import Ellipse, Arrow
from matplotlib.ticker import Locator, Formatter
from matplotlib import transforms as mtransforms
from matplotlib import rcParams

sentinel = object()


def plotHeatmap(fcsDF, x, y, vI=sentinel, bins=300, scale='linear', xscale='linear', yscale='linear', thresh=1000,
                aspect='auto', **kwargs):
    """
    Core plotting function of AliGater. Mainly intended to be called internally, but may be called directly.
    Only plots. No gating functionalities.
    
    **Parameters**
    
    fcsDF : pandas.DataFrame
        Flow data loaded in a pandas DataFrame.
    x, y : str
        Marker labels.
    vI : list-like, optional
        list-like index of events in the fcsDF that correspond to the parent population.
        Defaults to plotting all events in fcsDF.
    bins : int, optional, default: 300
        Resolution of the plotted heatmap.
    scale : str, optional, default: 'linear'
        Which scale to be used on both axes.
    xscale : str, optional, default: 'linear'
        Which scale to be used on the x-axis.
    yscale : str, optional, default: 'linear'
        Which scale to be used on the y-axis.
    T : int, optional, default: 1000
        If the threshold for linear-loglike transition for bilog or logicle scales.
    aspect : str
        Aspect of plotted heatmap. Passed on to matplotlib.pyplot.imshow()
    
    **Keyword arguments**
    
    cmap : matplotlib.colors.Colormap or str, default: 'jet'
        Color map to use. 
        Either string name of existing matplotlib colormap, or a colormap object.
        
    rcParams : matplotlib.rcParams
        Overrides rcParams with the passed rcParams object.
    
    mask_where : float,int, default : 0
        scalar of heatmap values to mask, these become white when plotted
    
    **Returns**

    fig, matplotlib.pyplot.Figure
        matplotlib Figure object
        
    ax. matplotlib.pyplot.Axes
        matplotlib axes object
        
    **Examples**

        None currently.
    """
    if vI is sentinel:
        vI = fcsDF.index
    elif len(vI) < 2:
        sys.stderr.write("Passed index contains no events\n")
        return None, None
    if not isinstance(bins, str) and len(vI) < bins:
        bins = len(vI)
    if scale.lower() == 'logicle':
        xscale = 'logicle'
        yscale = 'logicle'
    if scale.lower() == 'bilog':
        xscale = 'bilog'
        yscale = 'bilog'
    # Default x and y lims
    bYlim = False
    bXlim = False
    if 'xlim' in kwargs:
        if not isinstance(kwargs['xlim'], list):
            raise TypeError("if xlim is passed, it must be a list of float/int")
        elif not all(isinstance(i, (float, int)) for i in kwargs['xlim']):
            raise TypeError("Non float/int element encountered in xlim")
        else:
            xscale_limits = kwargs['xlim']
            if xscale.lower() == 'logicle':
                xscale_limits = logicleTransform(xscale_limits, thresh)
                bXlim = True
    if 'ylim' in kwargs:
        if not isinstance(kwargs['ylim'], list):
            raise TypeError("if ylim is passed, it must be a list of float/int")
        elif not all(isinstance(i, (float, int)) for i in kwargs['ylim']):
            raise TypeError("Non float/int element encountered in ylim")
        else:
            yscale_limits = kwargs['ylim']
            if yscale.lower() == 'logicle':
                yscale_limits = logicleTransform(yscale_limits, thresh)
                bYlim = True

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
        if not isinstance(cmap, str):
            collect_default = False
        else:
            collect_default = True
    else:
        collect_default = True
        cmap = 'jet'

    if 'rcParams' in kwargs:
        if not isinstance(kwargs['rcParams'], dict):
            raise TypeError("if rcParams is passed, it must be a dict")
        else:
            rcParams = kwargs['rcParams']
            custom_rcParams = True
    else:
        custom_rcParams = False

    if 'mask_where' in kwargs:
        mask_value = kwargs['mask_where']
        assert isinstance(mask_value, (float, int))
    else:
        mask_value = 0

    vX = getGatedVector(fcsDF, x, vI, return_type="nparray")
    vY = getGatedVector(fcsDF, y, vI, return_type="nparray")
    if len(vX) < 2 or len(vY) < 2:
        sys.stderr.write("Passed index contains no events\n")
        return None, None
    if custom_rcParams:
        plt.rcParams = rcParams
    else:
        plt.rcParams['figure.figsize'] = 10, 10
        plt.rcParams['image.cmap'] = cmap

        # extra
        plt.rcParams['font.size'] = 22
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        # plt.rcParams['label.size': 22]

    heatmap, xedges, yedges = getHeatmap(vX, vY, bins, scale, xscale, yscale, thresh)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    heatmap = np.ma.masked_where(heatmap <= mask_value, heatmap)

    figure = plt.figure()
    axes = figure.add_subplot(111)
    # matplotlib 3.2.x changed behaviour of interpolation
    # see https://github.com/matplotlib/matplotlib/issues/17722
    # and https://matplotlib.org/3.2.1/api/api_changes.html#default-image-interpolation
    if collect_default:
        cmap = plt.get_cmap().copy()
    cmap.set_bad(color='white')  # Zeroes should be white, not blue

    axes.imshow(heatmap.T, extent=extent, origin='lower', aspect=aspect, interpolation='none', cmap=cmap)

    axes.set_xlabel(x)
    axes.set_ylabel(y)

    if xscale.lower() == 'logicle':
        axes.xaxis.set_major_locator(logicleLocator(linCutOff=thresh))
        axes.xaxis.set_major_formatter(logicleFormatter(linCutOff=thresh))

    if yscale.lower() == 'logicle':
        axes.yaxis.set_major_locator(logicleLocator(linCutOff=thresh))
        axes.yaxis.set_major_formatter(logicleFormatter(linCutOff=thresh))

    if xscale.lower() == 'bilog':
        axes.xaxis.set_major_locator(BiLogLocator(linCutOff=thresh))
        axes.xaxis.set_major_formatter(BiLogFormatter(linCutOff=thresh))

    if yscale.lower() == 'bilog':
        axes.yaxis.set_major_locator(BiLogLocator(linCutOff=thresh))
        axes.yaxis.set_major_formatter(BiLogFormatter(linCutOff=thresh))

    if bXlim:
        axes.xaxis.set_xlim(left=xscale_limits[0], right=xscale_limits[1])

    if bYlim:
        axes.yaxis.set_xlim(left=yscale_limits[0], right=yscale_limits[1])

    return figure, axes


def getHeatmap(vX, vY, bins='auto', scale='linear', xscale='linear', yscale='linear', T=1000, normalize=False,
               xlim=None, ylim=None, range=None):
    if not any(isinstance(i, str) for i in [scale, xscale, yscale]):
        raise TypeError("scale, xscale, yscale must be specified as string, such as: 'linear', 'logicle'")
    if not all(i.lower() in ['linear', 'logicle', 'bilog'] for i in [scale, xscale, yscale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logicle'")
    if not isinstance(bins, (int, str)):
        raise TypeError("bins can only be either of int or str")
    if range is not None:
        if isinstance(range, list):
            if len(range) == 2:
                if not all(isinstance(i, (list)) for i in range):
                    AliGaterError("in getHeatmap, invalid dtype encountered in range, expected two list-likes")
                else:
                    if not all(isinstance(i, (float, int)) for i in range[0]) or not all(
                            isinstance(i, (float, int)) for i in range[1]):
                        AliGaterError("in getHeatmap,invalid dtype encountered in range")
                    else:
                        defaultRange = range
                        xRange = range[0]
                        yRange = range[1]
            else:
                AliGaterError("in getHeatmap, range must be list, found " + str(type(range)))
        else:
            AliGaterError("in getHeatmap, custom range passed but is not list, found type: " + str(type(range)))
    else:
        defaultRange = None
        xRange = None
        yRange = None

    if not len(vX) == len(vY):
        raise AliGaterError("in getHeatmap: ", "Coordinate vectors are of unequal length")
    if len(vX) == 0:
        raise AliGaterError("in getHeatmap: ", "Coordinate vectors are empty")

    if not isinstance(vX, np.ndarray):
        try:
            vX = np.asarray(vX)
        except:
            raise AliGaterError("in getHeatmap: ", "Couldn't coerce x-value vectors into numpy array format")
    if not isinstance(vY, np.ndarray):
        try:
            vY = np.asarray(vY)
        except:
            raise AliGaterError("in getHeatmap: ", "Couldn't coerce x-value vectors into numpy array format")
    index_mask = []
    for i in np.arange(len(vX) - 1, -1, -1):
        if xlim is not None:
            if vX[i] < xlim[0] or vX[i] > xlim[1]:
                index_mask.append(i)
                continue
            if vY[i] < ylim[0] or vY[i] > ylim[1]:
                index_mask.append(i)
    if len(index_mask) > 0:
        vX = np.delete(vX, index_mask)
        vY = np.delete(vY, index_mask)
        assert len(vX) == len(vY)

    if isinstance(bins, str):
        xbin_edges = np.histogram_bin_edges(vX, bins=bins)
        ybin_edges = np.histogram_bin_edges(vY, bins=bins)
    else:
        xbin_edges = bins
        ybin_edges = bins

    if scale.lower() == 'linear' and xscale.lower() == 'linear' and yscale.lower() == 'linear':
        return np.histogram2d(vX, vY, [xbin_edges, ybin_edges], normed=normalize, range=defaultRange)

    # if not linear probably just transform and calc edges after
    # attempt at fix, still some redundancy...
    t_xbin_edges = t_ybin_edges = None
    if scale.lower() != 'linear' or (xscale.lower() != 'linear' and yscale.lower() != 'linear'):
        # Solve case where scale != linear but xscale/yscale = linear
        if xscale.lower() == 'linear' and scale.lower() != 'linear':
            xscale = scale
        t_vX = transformWrapper(vX, scale=xscale, T=T)
        t_xbin_edges = np.histogram_bin_edges(t_vX, bins=bins)
        xbin_edges = inverseTransformWrapper(t_xbin_edges, scale=xscale, T=T)

        if yscale.lower() == 'linear' and scale.lower() != 'linear':
            yscale = scale
        t_vY = transformWrapper(vY, scale=yscale, T=T)
        t_ybin_edges = np.histogram_bin_edges(t_vY, bins=bins)
        ybin_edges = inverseTransformWrapper(t_ybin_edges, scale=yscale, T=T)
        return np.histogram2d(vX, vY, [xbin_edges, ybin_edges], normed=normalize, range=defaultRange)

    if xscale.lower() != 'linear' and scale.lower() == 'linear':
        t_vX = transformWrapper(vX, scale=xscale, T=T)
        t_xbin_edges = np.histogram_bin_edges(t_vX, bins=bins)
        xbin_edges = inverseTransformWrapper(t_xbin_edges, scale=xscale, T=T)

        ybin_edges = np.histogram_bin_edges(vY, bins=bins)

    if yscale.lower() != 'linear' and scale.lower() == 'linear':
        t_vY = transformWrapper(vY, scale=yscale, T=T)
        t_ybin_edges = np.histogram_bin_edges(t_vY, bins=bins)
        ybin_edges = inverseTransformWrapper(t_ybin_edges, scale=yscale, T=T)

        xbin_edges = np.histogram_bin_edges(vX, bins=bins)

    # print(ybin_edges)
    # print("\n\n")
    # print(xbin_edges)
    # print("\n\n")
    return np.histogram2d(vX, vY, [xbin_edges, ybin_edges], normed=normalize, range=defaultRange)

    # -------------------------DEPRECATED below---------------------------
    if scale == 'logicle' or (xscale == 'logicle' and yscale == 'logicle'):
        xBinEdges = logicleBin(vX, bins, T, xRange)
        yBinEdges = logicleBin(vY, bins, T, yRange)
        return np.histogram2d(vX, vY, [xBinEdges, yBinEdges], normed=normalize)
    if xscale == 'logicle':
        xBinEdges = logicleBin(vX, bins, T, xRange)
        return np.histogram2d(vX, vY, [xBinEdges, bins], normed=normalize)
    if yscale == 'logicle':
        yBinEdges = logicleBin(vY, bins, T, yRange)
        return np.histogram2d(vX, vY, [bins, yBinEdges], normed=normalize)
    if scale == 'bilog' or (xscale == 'bilog' and yscale == 'bilog'):
        xBinEdges = bilogBin(vX, bins, T, xRange)
        yBinEdges = bilogBin(vY, bins, T, yRange)
        # print("xBinEdges: ")
        # print(xBinEdges)
        # print("\n\n")
        # print("yBinEdges: ")
        # print(yBinEdges)
        return np.histogram2d(vX, vY, [xBinEdges, yBinEdges], normed=normalize)
    if xscale == 'bilog':
        xBinEdges = bilogBin(vX, bins, T, xRange)
        return np.histogram2d(vX, vY, [xBinEdges, bins], normed=normalize)
    if yscale == 'bilog':
        yBinEdges = bilogBin(vY, bins, T, yRange)
        return np.histogram2d(vX, vY, [bins, yBinEdges], normed=normalize)


def plot_flattened_heatmap(heatmap_array, nOfBins, mask=True):
    reshaped_array = heatmap_array.reshape(nOfBins, nOfBins)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if mask:
        heatmap = np.ma.masked_where(reshaped_array == 0, reshaped_array)
        cmap = plt.get_cmap().copy()
        cmap.set_bad(color='white')
    else:
        heatmap = reshaped_array
    ax.imshow(heatmap.T[::-1], cmap=cmap)
    plt.show()
    plt.close(fig)
    return None


def transformWrapper(vX, T, scale):
    """
    General function for converting values or arrays of values to AliGater scales; bilog and logicle.
    See inverseTransformWrapper to convert the other way around.
    
    **Parameters**
    
    vX, list-like or float/int
        value or values to convert.

    T, int/float
        Threshold for linear-log transition for bilog and logicle scales
    
    scale, str
        Scale to convert to; 'bilog' or 'logicle'
        
    **Returns** 
        If a scalar is passed, scalar
        If list like is passed, list

    **Examples**

    None currently.
    """
    result = None
    single_val = False

    if not isinstance(vX, (list, np.ndarray, tuple)):
        if isinstance(vX, (float, int)):
            vInput = np.asarray(vX).reshape(1, )
            single_val = True
        else:
            raise AliGaterError("in transformWrapper",
                                "invalid dType of passed vX, must be either a single float/int value or list/np.ndarray/tuple of float/int values")
    else:
        vInput = vX

    if not isinstance(vX, np.ndarray):
        try:
            vX = np.asarray(vX)
        except:
            raise AliGaterError("in transformWrapper: ", "Couldn't coerce input vector to numpy array format")

    if scale.lower() == 'logicle':
        result = logicleTransform(vInput, T)
    elif scale.lower() == 'bilog':
        result = bilogTransform(vInput, T)
    elif scale.lower() == 'linear':
        result = vX
    if result is None:
        raise
    if single_val:
        result = result[0]
    return result


def inverseTransformWrapper(vX, T, scale):
    """
    General function for converting values or arrays of values from AliGater scales; bilog and logicle back to linear values.
    See transformWrapper to convert into AliGater scales.
    
    **Parameters**
    
    vX, list-like or float/int
        value or values to convert.

    T, int/float
        Threshold for linear-log transition for bilog and logicle scales
    
    scale, str
        Scale to convert from; 'bilog' or 'logicle'
        
    **Returns** 
        If a scalar is passed, scalar
        If list like is passed, list

    **Examples**

    None currently.
    """
    result = None
    single_val = False
    if not isinstance(vX, (list, np.ndarray, tuple)):
        if isinstance(vX, (float, int)):
            vInput = np.asarray(vX).reshape(1, )
            single_val = True
        else:
            raise AliGaterError("in inverseTransformWrapper",
                                "invalid dType of passed vX, must be either a single float/int value or list/np.ndarray/tuple of float/int values")
    else:
        vInput = vX

    if not isinstance(vX, np.ndarray):
        try:
            vX = np.asarray(vX)
        except:
            raise AliGaterError("in inverseTransformWrapper: ", "Couldn't coerce input vector to numpy array format")

    if scale.lower() == 'logicle':
        result = inverselogicleTransform(vInput, T)
    elif scale.lower() == 'bilog':
        result = inverseBilogTransform(vInput, T)
    elif scale.lower() == 'linear':
        result = vX
    if result is None:
        raise
    if single_val:
        result = result[0]
    return result


def bilogBin(vX, bins, T, customRange=None):
    if customRange is not None:
        defaultRange = customRange
    else:
        defaultRange = [min(vX), max(vX)]
    transformedRange = bilogTransform(defaultRange, T)
    transformedBinEdges = np.linspace(transformedRange[0], transformedRange[1], bins + 1)
    return inverseBilogTransform(transformedBinEdges, T)


def bilogTransform(a, T):
    vA = np.asarray(a, dtype=np.float64, order='C')
    tA = __vectorBilogTransform(vA, np.float64(T))
    return tA

    # old python implementation, moved to AGCythonUtils
    # tA = np.empty_like(a).astype(float)
    # a_idx=0
    # while a_idx < len(a):
    #     if a[a_idx] >= T:
    #        tA[a_idx] = np.log(10 * a[a_idx] / T)/np.log(10)
    #     elif a[a_idx] < T and a[a_idx] > -T:
    #         tA[a_idx] = (a[a_idx]/T  + np.log(10) - 1) / np.log(10)
    #     else:
    #         tA[a_idx] = -np.log(10 * abs(a[a_idx]) / T) / np.log(10)+1.13141103619349642 #This shift ensures that the transformed coordinates are continous, important for bins and plotting
    #     a_idx+=1
    # return tA


def inverseBilogTransform(a, T):
    vA = np.asarray(a, dtype=np.float64, order='C')
    invA = __vectorInverseBilogTransform(vA, np.float64(T))
    return invA

    # old python implementation, moved to AGCythonUtils
    # invA=np.empty_like(a).astype(float)
    # a_idx=0
    # while a_idx < len(a):
    #     if a[a_idx] >= 1.0: #transformed linCutOff, always 1.0 at T; np.log(10 * linCutOff / linCutOff)/np.log(10) -> np.log(10)/np.log(10) = 1 
    #         invA[a_idx] = T*np.exp(np.log(10)*a[a_idx])/10
    #     elif a[a_idx] <= 0.13141103619349642: #This is (np.log(10)-2)/np.log(10) I.e. the linear scale value at X=-T
    #         tmpX=a[a_idx]-1.13141103619349642 #This shift ensures that the transformed coordinates are continous, important for bins and plotting
    #         invA[a_idx] = -T*np.exp(np.log(10)*-tmpX)/10
    #     else:
    #         invA[a_idx] = T * (np.log(10)*a[a_idx] - np.log(10) + 1)
    #     a_idx+=1
    # return invA


def logicleBin(vX, bins, T, customRange=None):
    if customRange is not None:
        defaultRange = customRange
    else:
        defaultRange = [min(vX), max(vX)]
    transformedRange = logicleTransform(defaultRange, T)
    transformedBinEdges = np.linspace(transformedRange[0], transformedRange[1], bins + 1)
    return inverselogicleTransform(transformedBinEdges, T)


def logicleTransform(a, linCutOff):
    vA = np.asarray(a, dtype=np.float64, order='C')

    tA = __vectorlogicleTransform(vA, np.float64(linCutOff))

    return tA

    # old python implementation, moved to AGCythonUtils
    # tA = np.empty_like(a).astype(float)
    # a_idx=0
    # while a_idx < len(a):
    #     if a[a_idx] >= linCutOff:
    #        tA[a_idx] = np.log(10 * a[a_idx] / linCutOff)/np.log(10)
    #     else:
    #         tA[a_idx] = (a[a_idx]/linCutOff + np.log(10.0) - 1)/np.log(10)
    #     a_idx+=1
    # return tA


def inverselogicleTransform(a, linCutOff):
    vA = np.asarray(a, dtype=np.float64, order='C')
    invA = __vectorInverselogicleTransform(vA, np.float64(linCutOff))

    return invA
    # old python implementation, moved to AGCythonUtils
    # invA=np.empty_like(a).astype(float)
    # a_idx=0
    # while a_idx < len(a):
    #     if a[a_idx] >= 1.0: #transformed linCutOff, always 1.0 at T; np.log(10 * linCutOff / linCutOff)/np.log(10) -> np.log(10)/np.log(10) = 1 
    #         invA[a_idx] = linCutOff*np.exp(np.log(10)*a[a_idx])/10
    #         #invA[a_idx]= (np.exp(a[a_idx])+10)*linCutOff/10
    #     else:
    #         invA[a_idx] = linCutOff*(np.log(10.0)*a[a_idx] - np.log(10.0) + 1)
    #     a_idx+=1
    # return invA


def addAxLine(fig, ax, pos, orientation, size=2, scale='linear', xscale='linear', yscale='linear', T=1000):
    if not all(i in ['linear', 'logicle', 'bilog'] for i in [scale, xscale, yscale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logicle', 'bilog'")
    if orientation.lower() == 'vertical':
        if scale.lower() != 'linear' or xscale.lower() != 'linear':
            lims = ax.get_xlim()
            vmin = lims[0]
            vmax = lims[1]
            if scale.lower() == 'logicle' or xscale.lower() == 'logicle':
                pos = convertTologiclePlotCoordinate(pos, vmin, vmax, T)
            if scale.lower() == 'bilog' or xscale.lower() == 'bilog':
                pos = convertToBiLogPlotCoordinate(pos, vmin, vmax, T)
        ax.axvline(pos, c='r')
    else:
        if scale.lower() != 'linear' or yscale.lower() != 'linear':
            lims = ax.get_ylim()
            vmin = lims[0]
            vmax = lims[1]
            if scale == 'logicle' or yscale.lower() == 'logicle':
                pos = convertTologiclePlotCoordinate(pos, vmin, vmax, T)
            if scale.lower() == 'bilog' or yscale.lower() == 'bilog':
                pos = convertToBiLogPlotCoordinate(pos, vmin, vmax, T)
        ax.axhline(pos, c='r')
    return fig


def addLine(fig, ax, lStartCoordinate, lEndCoordinate, size=2, scale='linear', T=1000):
    if not scale.lower() in ['linear', 'logicle', 'bilog']:
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logicle', 'bilog'")
    if scale.lower() == 'logicle':
        view = ax.xaxis.get_view_interval()
        xCoordinates = convertTologiclePlotCoordinates([lStartCoordinate[0], lEndCoordinate[0]], vmin=view[0],
                                                       vmax=view[1], T=T)
        view = ax.yaxis.get_view_interval()
        yCoordinates = convertTologiclePlotCoordinates([lStartCoordinate[1], lEndCoordinate[1]], vmin=view[0],
                                                       vmax=view[1], T=T)
        lStartCoordinate = [xCoordinates[0], yCoordinates[0]]
        lEndCoordinate = [xCoordinates[1], yCoordinates[1]]
    if scale.lower() == 'bilog':
        view = ax.xaxis.get_view_interval()
        xCoordinates = convertToBiLogPlotCoordinates([lStartCoordinate[0], lEndCoordinate[0]], vmin=view[0],
                                                     vmax=view[1], T=T)
        view = ax.yaxis.get_view_interval()
        yCoordinates = convertToBiLogPlotCoordinates([lStartCoordinate[1], lEndCoordinate[1]], vmin=view[0],
                                                     vmax=view[1], T=T)
        lStartCoordinate = [xCoordinates[0], yCoordinates[0]]
        lEndCoordinate = [xCoordinates[1], yCoordinates[1]]
    ax.plot([lStartCoordinate[0], lEndCoordinate[0]], [lStartCoordinate[1], lEndCoordinate[1]], color='r',
            linestyle='-', linewidth=size, figure=fig)
    return fig, ax


def addArrow(fig, ax, lStartCoordinate, lEndCoordinate, size=5000):
    arrow = Arrow(lStartCoordinate[0], lStartCoordinate[1], lEndCoordinate[0] - lStartCoordinate[0],
                  lEndCoordinate[1] - lStartCoordinate[1], width=size, transform=ax.transAxes, head_width=size,
                  head_length=size, fc='r', ec='r')
    # ax.arrow(lStartCoordinate[0], lStartCoordinate[1], lEndCoordinate[0]-lStartCoordinate[0], lEndCoordinate[1]-lStartCoordinate[1], head_width=size, head_length=size, fc='r', ec='r')
    ax.add_patch(arrow)
    return fig


def draw_ellipse(position, covariance, sigma=2, ax=None, **kwargs):
    if agconf.execMode in ["jupyter", "ipython"]:
        plot = True
    else:
        plot = False
    if plot:
        ax = ax or plt.gca();

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = np.sqrt(s) * sigma
    else:
        angle = 0
        width = height = np.sqrt(covariance) * sigma
    # Note width, height here is the full width and height and not the semiaxis length
    # Draw the Ellipse
    if plot:
        ax.add_patch(Ellipse(position, width, height,
                             angle, **kwargs));
    return width, height, angle


def plot_gmm(fcsDF, xCol, yCol, vI, gmm, sigma, ax):
    if agconf.execMode in ["jupyter", "ipython"]:
        plot = True
    else:
        plot = False
    ax = ax or plt.gca()
    vEllipses = []
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        width, height, angle = draw_ellipse(pos, covar, sigma, fill=False, edgecolor='#FF0000', linestyle='dashed');
        vEllipses.append([pos, width, height, angle])
    if plot:
        plt.show();
    return vEllipses


def plot_densityFunc(fcsDF, xCol, vI=sentinel, sigma=3, bins=300, scale='linear', T=1000, *args, **kwargs):
    """
    General function for converting values or arrays of values from AliGater scales; bilog and logicle back to linear values.
    See transformWrapper to convert into AliGater scales.
    
    **Parameters**
    
    vX, list-like or float/int
        value or values to convert.
    
    T, int/float
        Threshold for linear-log transition for bilog and logicle scales
    
    scale, str
        Scale to convert from; 'bilog' or 'logicle'
    
    **Returns** 
        If a scalar is passed, scalar
        If list like is passed, list
    
    **Examples**
    
    None currently.
    """
    if xCol not in fcsDF.columns:
        raise TypeError("Specified gate not in dataframe, check spelling or control your dataframe.columns labels")
    if vI is sentinel:
        vI = fcsDF.index
    elif len(vI) == 0:
        sys.stderr.write("Passed index contains no events\n")
        return None
    if not all(i in ['linear', 'logicle', 'bilog'] for i in [scale]):
        raise TypeError("scale, xscale, yscale can only be either of: 'linear', 'logicle', 'bilog'")
    if not isinstance(sigma, (float, int)):
        raise AliGaterError("Sigma must be float or int, found: " + str(type(sigma)), "in plot_densityFunc")
    if 'sigma' in kwargs:
        if not isinstance(kwargs['sigma'], (float, int)):
            raise AliGaterError("Sigma must be float or int, found: " + str(type(sigma)), "in plot_densityFunc")
        else:
            sigma = kwargs['sigma']

    data = getGatedVector(fcsDF, xCol, vI, return_type="nparray")

    if isinstance(bins, int):
        if len(vI) < bins:
            sys.stderr.write("Fewer events than bins, readjusting number of bins\n")
            bins = len(vI)
    elif bins == 'auto':
        if scale.lower() != 'linear':
            t_data = transformWrapper(data, T=T, scale=scale)
        else:
            t_data = data
        bins = __autoBinCount(t_data)
    else:
        raise AliGaterError("bins must be integer or string 'auto'", "in plot_densityFunc")

    if scale == 'logicle':
        BinEdges = logicleBin(data, bins, T)
        histo = np.histogram(data, BinEdges)
    elif scale == 'bilog':
        BinEdges = bilogBin(data, bins, T)
        histo = np.histogram(data, BinEdges)
    else:
        histo = np.histogram(data, bins)
    vHisto = np.linspace(min(histo[1]), max(histo[1]), bins)
    smoothedHisto = gaussian_filter1d(histo[0].astype(float), sigma)
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.plot(vHisto, smoothedHisto, label="pdf for " + str(xCol) + "\nsigma: " + str(sigma))
    axes.legend(loc='upper right', shadow=True, fontsize='medium')
    if scale.lower() != 'linear':
        axes.set_xlim(left=min(data), right=max(data))
        if scale.lower() == 'logicle':
            axes.xaxis.set_major_locator(logicleLocator(linCutOff=T))
            axes.xaxis.set_major_formatter(logicleFormatter(linCutOff=T))
        if scale.lower() == 'bilog':
            axes.xaxis.set_major_locator(BiLogLocator(linCutOff=T))
            axes.xaxis.set_major_formatter(BiLogFormatter(linCutOff=T))
    return figure, axes


def __autoBinCount(data):
    # Internal function that mimics numpus numpy.histogram_bin_edges functionality to guess appropriate number of bins
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html
    data_IQR = iqr(data)
    n = len(data)
    fd_h = 2 * (data_IQR / (np.power(n, (1 / 3))))  # Freedman Diaconis Estimator
    fd_bins = np.round(np.ceil((max(data) - min(data)) / fd_h))  # np.round(np.ceil(range / h))
    s_bins = np.log2(n) + 1  # Sturges estimator
    bins = int(max([fd_bins, s_bins]))

    return bins


def imagePCA_cluster(imlist, samplelist, nOfComponents=2):
    immatrix = np.array([im.flatten() for im in imlist], 'f')

    # Check for nan elements in matrix
    if np.isnan(immatrix).any():
        array_has_nan = np.array([np.isnan(arr).any() for arr in immatrix])
        removed_images = samplelist[array_has_nan]
        imlist = imlist[~array_has_nan]
        samplelist = samplelist[~array_has_nan]
        n_of_nan = array_has_nan.sum()
        reportStr = str(n_of_nan) + " samples had invalid images and where removed:\n" + "\n".join(
            removed_images) + "\n"
        sys.stderr.write(reportStr)
        immatrix = np.array([im.flatten() for im in imlist], 'f')

    if immatrix.shape[0] == 0:
        reportStr = "No data in passed image matrix\n"
        sys.stderr.write(reportStr)
        return None
    if immatrix.shape[0] < nOfComponents:
        reportStr = "WARNING: fewer samples than requested components for PC analysis, adjusting\n"
        sys.stderr.write(reportStr)
        nOfComponents = immatrix.shape[0]
    pca_obj = PCA(n_components=nOfComponents)
    pca_obj.fit(immatrix)
    projection_d = pca_obj.transform(immatrix)
    projection_d_df = pd.DataFrame(projection_d)
    projection_d_df.index = samplelist
    columnNames = []
    for i in np.arange(1, nOfComponents + 1, 1):
        columnStr = "PC" + str(i)
        columnNames.append(columnStr)
    projection_d_df.columns = columnNames
    reportStr = "PCs explained variance: \n" + str(
        pca_obj.explained_variance_ratio_) + "\nTotal visual variation explained: " + str(
        sum(pca_obj.explained_variance_ratio_)) + "\n"
    sys.stderr.write(reportStr)
    # center the coordinate system on the mean of each PC
    projection_d_df = projection_d_df - projection_d_df.mean()

    projection_d_df['length'] = np.sqrt(np.square(projection_d_df).sum(axis=1))
    projection_d_df.sort_values(by='length', inplace=True)

    return projection_d_df


def imagePCA(imlist):
    """
    Perform Principal Component Analysis of downsampled heatmap images, plots results.
    Takes a list-like of heatmap images, flattens them and calls image_pca.
    
    **Parameters**
    
    X, list-like of list-like
        Matrix with image data stored
        
    **Returns** 
   
    None

    **Examples**

    None currently.
    """
    m = 16
    n = 16
    # create matrix to store all flattened images
    immatrix = np.array([im.flatten() for im in imlist], 'f')

    # perform PCA
    V, S, immean = image_pca(immatrix)
    # show some images (mean and 7 first modes)
    plt.figure(figsize=(20, 30))
    plt.subplot(2, 4, 1)
    plt.imshow(immean.reshape(m, n))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(V[i].reshape(m, n).T[::-1], cmap="bwr")
    plt.show()


def image_pca(X):
    # Based on Stack Overflow discussion and code here
    # https://math.stackexchange.com/questions/409239/compute-pca-with-this-useful-trick
    """
    Principal Component Analysis of flattened heatmap images, main purpose is to be called internally by imagePCA
    
    **Parameters**
    
    X, list-like
        List-like matrix with image data stored as flattened arrays in rows.
        
   **Returns** 
   
    List-like    
        Projection matrix (with important dimensions first)
    Float
        Variance
    Float
        Mean

    **Examples**

    None currently.
    """

    # get dimensions
    try:
        num_data, dim = X.shape
    except ValueError:
        sys.stderr.write("WARNING in image_pca: input matrix invalid\n")
        return None, None, None
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick used
        M = np.dot(X, X.T)  # covariance matrix
        e, EV = np.linalg.eigh(M)  # eigenvalues and eigenvectors
        tmp = np.dot(X.T, EV).T  # this is the compact trick
        V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]  # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V, S, mean_X


# From Ticker.py
def decade_down(x, base=10):
    'floor x to the nearest lower decade'
    if x == 0.0:
        return -base
    lx = np.floor(np.log(x) / np.log(base))
    return base ** lx


# From Ticker.py
def decade_up(x, base=10):
    'ceil x to the nearest higher decade'
    if x == 0.0:
        return base
    lx = np.ceil(np.log(x) / np.log(base))
    return base ** lx


def is_decade(x, base=10):
    if not np.isfinite(x):
        return False
    if x == 0.0:
        return True
    lx = np.log(np.abs(x)) / np.log(base)
    return is_close_to_int(lx)


# From Ticker.py
def is_close_to_int(x):
    if not np.isfinite(x):
        return False
    return abs(x - nearest_int(x)) < 1e-10


# From Ticker.py
def nearest_int(x):
    if x == 0:
        return int(0)
    elif x > 0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)


def convertToBiLogPlotCoordinates(Ticlocs, vmin, vmax, T):
    actualRange = vmax - vmin
    tMinMax = bilogTransform([vmin, vmax], T)
    transformedRange = tMinMax[1] - tMinMax[0]
    tTiclocs = bilogTransform(Ticlocs, T)
    plotTics = []
    for tTic in tTiclocs:
        plotTic = (tTic - tMinMax[0]) / transformedRange * actualRange + vmin
        plotTics.append(plotTic)
    assert len(tTiclocs) == len(Ticlocs)
    return plotTics


def convertToBiLogPlotCoordinate(Ticloc, vmin, vmax, T):
    actualRange = vmax - vmin
    tMinMax = bilogTransform([vmin, vmax], T)
    transformedRange = tMinMax[1] - tMinMax[0]
    tTicloc = bilogTransform([Ticloc], T)[0]
    plotTic = (tTicloc - tMinMax[0]) / transformedRange * actualRange + vmin
    return plotTic


def invertBiLogPlotcoordinates(plotTics, vmin, vmax, T):
    actualRange = vmax - vmin
    tMinMax = bilogTransform([vmin, vmax], T)
    transformedRange = tMinMax[1] - tMinMax[0]
    invPlotTics = []
    for tTic in plotTics:
        invPlotTic = (tTic - vmin) / actualRange * transformedRange + tMinMax[0]
        invPlotTics.append(invPlotTic)
    result = inverseBilogTransform(invPlotTics, T)
    return result


def invertBiLogPlotcoordinate(plotTic, vmin, vmax, T):
    actualRange = vmax - vmin
    tMinMax = bilogTransform([vmin, vmax], T)
    transformedRange = tMinMax[1] - tMinMax[0]
    invPlotTic = (plotTic - vmin) / actualRange * transformedRange + tMinMax[0]
    result = inverseBilogTransform([invPlotTic], T)[0]
    return result


def convertTologiclePlotCoordinates(Ticlocs, vmin, vmax, T):
    actualRange = vmax - vmin
    tMinMax = logicleTransform([vmin, vmax], T)
    transformedRange = tMinMax[1] - tMinMax[0]
    tTiclocs = logicleTransform(Ticlocs, T)
    plotTics = []
    for tTic in tTiclocs:
        plotTic = (tTic - tMinMax[0]) / transformedRange * actualRange + vmin
        plotTics.append(plotTic)
    assert len(tTiclocs) == len(Ticlocs)
    return plotTics


def convertTologiclePlotCoordinate(Ticloc, vmin, vmax, T):
    actualRange = vmax - vmin
    tMinMax = logicleTransform([vmin, vmax], T)
    transformedRange = tMinMax[1] - tMinMax[0]
    tTicloc = logicleTransform([Ticloc], T)[0]
    plotTic = (tTicloc - tMinMax[0]) / transformedRange * actualRange + vmin
    return plotTic


def invertlogiclePlotcoordinates(plotTics, vmin, vmax, T):
    actualRange = vmax - vmin
    tMinMax = logicleTransform([vmin, vmax], T)
    transformedRange = tMinMax[1] - tMinMax[0]
    invPlotTics = []
    for tTic in plotTics:
        invPlotTic = (tTic - vmin) / actualRange * transformedRange + tMinMax[0]
        invPlotTics.append(invPlotTic)
    result = inverselogicleTransform(invPlotTics, T)
    return result


def invertlogiclePlotcoordinate(plotTic, vmin, vmax, T):
    actualRange = vmax - vmin
    tMinMax = logicleTransform([vmin, vmax], T)
    transformedRange = tMinMax[1] - tMinMax[0]
    invPlotTic = (plotTic - vmin) / actualRange * transformedRange + tMinMax[0]
    result = inverselogicleTransform([invPlotTic], T)[0]
    return result


class logicleLocator(Locator):
    # Modified from matplotlibs LogLocator
    # https://matplotlib.org/3.1.1/_modules/matplotlib/ticker.html#LogLocator
    """
    Determine the tick locations for logicle axes based on LogLocator. Only locates and formats tics for the plot view.
    Transform of underlying data and heatmap is handled outside matplotlib.
    Hacked version of LogLogator that covers normal usecases of the logicle scale
    Only defined with ticlocations for data in range -1 000 000 < x 
    """

    def __init__(self, linCutOff=1000, subs=(1.0,), numdecs=4, numticks=None):
        """
        Place ticks on the locations : subs[j] * base**i
        Parameters
        ----------
        subs : None, string, or sequence of float, optional, default (1.0,)
            Gives the multiples of integer powers of the base at which
            to place ticks.  The default places ticks only at
            integer powers of the base.
            The permitted string values are ``'auto'`` and ``'all'``,
            both of which use an algorithm based on the axis view
            limits to determine whether and how to put ticks between
            integer powers of the base.  With ``'auto'``, ticks are
            placed only between integer powers; with ``'all'``, the
            integer powers are included.  A value of None is
            equivalent to ``'auto'``.
        """
        if numticks is None:
            if rcParams['_internal.classic_mode']:
                numticks = 15
            else:
                numticks = 'auto'

        self._base = 10.0  # np.exp(1)
        self.subs(subs)
        self.numdecs = numdecs
        self.numticks = numticks

        if linCutOff > 10000:
            raise AliGaterError("in logicleLocator: ", "linear-log scale threshold can max be 10000")
        if linCutOff <= 0:
            raise AliGaterError("in logicleLocator: ", "linear-log scale threshold must be > 0")
        self.T = linCutOff

    def set_params(self, subs=None, numdecs=4, numticks=None):
        """Set parameters within this locator."""
        if subs is not None:
            self.subs(subs)
        if numdecs is not None:
            self.numdecs = numdecs
        if numticks is not None:
            self.numticks = numticks

    # FIXME: these base and subs functions are contrary to our
    # usual and desired API.

    def subs(self, subs):
        """
        set the minor ticks for the log scaling every base**i*subs[j]
        """
        if subs is None:  # consistency with previous bad API
            self._subs = 'auto'
        elif isinstance(subs, six.string_types):
            if subs not in ('all', 'auto'):
                raise ValueError("A subs string must be 'all' or 'auto'; "
                                 "found '%s'." % subs)
            self._subs = subs
        else:
            self._subs = np.asarray(subs, dtype=float)

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.view_limits()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if self.numticks == 'auto':
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks

        if vmax < vmin:
            vmin, vmax = vmax, vmin
        # If vmax-vmin flipped, correct it

        # How many decs in the log part?
        log_vmin = math.log(self.T) / math.log(self._base)
        try:
            log_vmax = math.log(vmax) / math.log(
                self._base)  # If empty input in log-span this can lead to math domain error. Return small default span in that case
        except ValueError:
            log_vmax = log_vmin + 1.0

        numdec = math.floor(log_vmax) - math.ceil(log_vmin)
        ticklocs = self._base ** numdec  # Base ** decades

        if numdec > 10:
            subs = np.array([1.0])
        else:
            subs = np.arange(1.0, self._base)  # (1.0, Base)
        stride = 1

        if rcParams['_internal.classic_mode']:
            # Leave the bug left over from the PY2-PY3 transition.
            while numdec / stride + 1 > numticks:
                stride += 1
        else:
            while numdec // stride + 1 > numticks:
                stride += 1

        # Does subs include anything other than 1?
        have_subs = len(subs) > 1 or (len(subs == 1) and subs[0] != 1.0)
        decades = np.arange(math.ceil(log_vmin) - stride, math.ceil(log_vmax) + 2 * stride, stride)

        if have_subs:
            ticklocs = []
            if stride == 1:
                for decadeStart in self._base ** decades:
                    ticklocs.extend(subs * decadeStart)
        else:
            ticklocs = self._base ** decades  # Base ** decades

        # Now we have number of tics and decs in the log part

        tmpTicLoc = []
        if vmin < -100000:
            tmpTicLoc.extend(np.arange(-1000000, -90000, 100000))

        if vmin < -10000:
            tmpTicLoc.extend(np.arange(-100000, -9000, 10000))

        if vmin < -1000:
            tmpTicLoc.extend(np.arange(-10000, -900, 1000))

        if vmin < 0:
            tmpTicLoc.extend(np.arange(-1000, 1, 200))

        Ticlocs = list(set(np.clip(tmpTicLoc, vmin, self.T)))
        Ticlocs = list(np.sort(Ticlocs))
        if vmax >= 0:
            tmpTicLoc.extend(np.arange(0, self.T, 1000))
            Ticlocs.extend(tmpTicLoc)

        # ticklocs.extend(Ticlocs)
        Ticlocs.extend(ticklocs)
        clip_Ticlocs = np.sort(list(set(np.clip(Ticlocs, vmin, vmax))))
        Ticlocs = convertTologiclePlotCoordinates(np.sort(clip_Ticlocs), vmin, vmax, self.T)
        # ADD HOC POSSIBLY
        Ticlocs = Ticlocs[1:-1]
        # Ticlocs=convertTologiclePlotCoordinates(Ticlocs, vmin, vmax, self.T)
        return self.raise_if_exceeds(np.asarray(Ticlocs))

    def view_limits(self, vmin=None, vmax=None):
        'Try to choose the view limits intelligently'
        vmin, vmax = self.axis.get_view_interval()
        return vmin, vmax


class logicleFormatter(Formatter):
    # Modified from matplotlibs LogFormatter
    # https://matplotlib.org/3.1.1/_modules/matplotlib/ticker.html#LogFormatter
    """
    Base class for formatting ticks on a logicle scale. Only locates and formats tics for the plot view.
    Transform of underlying data and heatmap is handled outside matplotlib.
    Modfied version of LogFormatter that covers normal usecases of the logicle scale
    Only defined with formatting ticlabels for data in range -1 000 000 < x
    The passed parameters only affect plotting of the log-part of the scale
    
    Parameters
    ----------
    labelOnlyBase : bool, optional, default: False
        If True, label ticks only at integer powers of base.
        This is normally True for major ticks and False for
        minor ticks.
    minor_thresholds : (subset, all), optional, default: (1, 0.4)
        If labelOnlyBase is False, these two numbers control
        the labeling of ticks that are not at integer powers of
        base; normally these are the minor ticks. The controlling
        parameter is the log of the axis data range.  In the typical
        case where base is 10 it is the number of decades spanned
        by the axis, so we can call it 'numdec'. If ``numdec <= all``,
        all minor ticks will be labeled.  If ``all < numdec <= subset``,
        then only a subset of minor ticks will be labeled, so as to
        avoid crowding. If ``numdec > subset`` then no minor ticks will
        be labeled.
    linthresh : float, optional, default: 1000
        The threshold for the logicle scale change from linear-like to log-like scaling
 
    Notes
    -----
    The `set_locs` method must be called to enable the subsetting
    logic controlled by the ``minor_thresholds`` parameter.
    In some cases such as the colorbar, there is no distinction between
    major and minor ticks; the tick locations might be set manually,
    or by a locator that puts ticks at integer powers of base and
    at intermediate locations.  For this situation, disable the
    minor_thresholds logic by using ``minor_thresholds=(np.inf, np.inf)``,
    so that all ticks will be labeled.
    To disable labeling of minor ticks when 'labelOnlyBase' is False,
    use ``minor_thresholds=(0, 0)``.  This is the default for the
    "classic" style.
    Examples
    --------
    To label a subset of minor ticks when the view limits span up
    to 2 decades, and all of the ticks when zoomed in to 0.5 decades
    or less, use ``minor_thresholds=(2, 0.5)``.
    To label all minor ticks when the view limits span up to 1.5
    decades, use ``minor_thresholds=(1.5, 1.5)``.
    """

    def __init__(self, labelOnlyBase=True,
                 minor_thresholds=None,
                 linCutOff=1000):

        self.labelOnlyBase = labelOnlyBase
        if minor_thresholds is None:
            if rcParams['_internal.classic_mode']:
                minor_thresholds = (0, 0)
            else:
                minor_thresholds = (1, 0.4)
        self.minor_thresholds = minor_thresholds
        self._sublabels = None
        self._linthresh = linCutOff
        self._base = np.exp(1)

    def _num_to_string(self, x, vmin, vmax):
        x = round(x, 0)
        s = self.pprint_val(x, vmax - vmin)
        return s

    def __call__(self, x, pos=None):
        """
        Return the format for tick val `x`.
        """
        if x == 0.0:  # Symlog
            return '0'
        vmin, vmax = self.axis.get_view_interval()

        tx = invertlogiclePlotcoordinate(x, vmin, vmax, self._linthresh)

        if tx > self._linthresh + 1:
            fx = math.log(tx) / math.log(10.0)
            is_x_decade = is_close_to_int(fx)
            exponent = np.round(fx) if is_x_decade else np.floor(fx)
            coeff = np.round(x / 10.0 ** exponent)

            if self.labelOnlyBase and not is_x_decade:
                return ''
            if self._sublabels is not None and coeff not in self._sublabels:
                return ''
        else:
            # Manually define acceptable negative values
            accepted_range = list(np.arange(-1000, 1001, 500))
            accepted_range.extend(np.arange(-10000, -1000, 5000))
            accepted_range.extend(np.arange(-100000, -9000, 10000))
            accepted_range.extend(np.arange(-1000000, -90000, 100000))
            if not np.round(tx) in accepted_range:
                return ''
        s = self._num_to_string(tx, vmin, vmax)
        return self.fix_minus(s)

    def pprint_val(self, x, d):
        # If the number is at or below the set lin-cutoff (_lintrehsh)
        # Print it as an int

        if x <= self._linthresh + 1:
            return '%d' % x

        fmt = '%1.3e'
        s = fmt % x
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            if exponent:
                if float(mantissa) > 1:
                    s = '$%s*10^{%d}$' % (mantissa, exponent)
                else:
                    s = '$%s0^{%d}$' % (mantissa, exponent)
            else:
                s = mantissa
        else:
            s = s.rstrip('0').rstrip('.')
        return s


class BiLogLocator(Locator):
    """
    Modified version of SymmetricalLogLocator. Only locates and formats tics for the plot view.
    Transform of underlying data and heatmap is handled outside matplotlib classes.
    Determine the tick locations for symmetric log axes
    """

    def __init__(self, subs=(1.0,), linCutOff=100):
        """
        place ticks on the location= base**i*subs[j]
        """
        self._base = 10  # np.exp(1)
        if isinstance(linCutOff, (float, int)):
            self.T = linCutOff
        else:
            raise AliGaterError("in BiLogLocator: ", "linthresh must be float/int. Found: " + str(type(linCutOff)))
        if subs is None:
            self._subs = [1.0]
        else:
            self._subs = subs
        self.numticks = 15

    def set_params(self, subs=None, numticks=None):
        """Set parameters within this locator."""
        if numticks is not None:
            self.numticks = numticks
        if subs is not None:
            self._subs = subs

    def __call__(self):
        'Return the locations of the ticks'
        # Note, these are untransformed coordinates
        # if view limits are to be chosen intelligently it must be done prior to heatmap creation,
        # thus at the level of plotheatmap. Before any ticformatting is made.
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        b = self._base
        t = self.T

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # The domain is divided into three sections, only some of
        # which may actually be present.
        #
        # <======== -t ==0== t ========>
        # aaaaaaaaa    bbbbb   ccccccccc
        #
        # a) and c) will have ticks at integral log positions.  The
        # number of ticks needs to be reduced if there are more
        # than self.numticks of them.
        #
        # b) has a tick at 0 and only 0 (we assume t is a small
        # number, and the linear segment is just an implementation
        # detail and not interesting.)
        #
        # We could also add ticks at t, but that seems to usually be
        # uninteresting.
        #
        # "simple" mode is when the range falls entirely within (-t,
        # t) -- it should just display (vmin, 0, vmax)

        has_a = has_b = has_c = False
        if vmin < -t:
            has_a = True
            if vmax > -t:
                has_b = True
                if vmax > t:
                    has_c = True
        elif vmin < 0:
            if vmax > 0:
                has_b = True
                if vmax > t:
                    has_c = True
            else:
                return [vmin, vmax]
        elif vmin < t:
            if vmax > t:
                has_b = True
                has_c = True
            else:
                return [vmin, vmax]
        else:
            has_c = True

        def get_log_range(lo, hi):
            lo = np.floor(np.log(lo) / np.log(b))
            hi = np.ceil(np.log(hi) / np.log(b))
            return lo, hi

        # First, calculate all the ranges, so we can determine striding
        if has_a:
            if has_b:
                a_range = get_log_range(t, -vmin + 1)
            else:
                a_range = get_log_range(-vmax, -vmin + 1)
        else:
            a_range = (0, 0)

        if has_c:
            if has_b:
                c_range = get_log_range(t, vmax + 1)
            else:
                c_range = get_log_range(vmin, vmax + 1)
        else:
            c_range = (0, 0)

        total_ticks = (a_range[1] - a_range[0]) + (c_range[1] - c_range[0])
        if has_b:
            total_ticks += 1
        stride = max(total_ticks // (self.numticks - 1), 1)

        decades = []
        if has_a:
            decades.extend(-1 * (b ** (np.arange(a_range[0], a_range[1],
                                                 stride)[::-1])))

        if has_b:
            decades.append(0.0)

        if has_c:
            decades.extend(b ** (np.arange(c_range[0], c_range[1], stride)))

        # Add the subticks if requested

        subs = np.arange(2.0, b)

        ticklocs = []
        for decade in decades:
            if decade == 0:
                ticklocs.append(decade)
            else:
                ticklocs.append(decade)
                if len(subs) > 1:
                    ticklocs.extend(subs * decade)

        clip_Ticlocs = np.sort(list(set(np.clip(ticklocs, vmin, vmax))))
        Ticlocs = convertToBiLogPlotCoordinates(np.sort(clip_Ticlocs), vmin, vmax, self.T)
        # dont want extra tic at min and max val
        Ticlocs = Ticlocs[1:-1]

        return self.raise_if_exceeds(np.array(Ticlocs))

    def view_limits(self, vmin=None, vmax=None):
        'Try to choose the view limits intelligently'
        # if view limits are to be chosen intelligently it must be done prior to heatmap creation,
        # thus at the level of plotheatmap. Before any ticformatting is made.
        vmin, vmax = self.axis.get_view_interval()
        return vmin, vmax


class BiLogFormatter(Formatter):
    """
    Base class for formatting ticks on a log or symlog scale.
    It may be instantiated directly, or subclassed.
    
    **Parameters**
    
    base : float, optional, default: 10.
        Base of the logarithm used in all calculations.
    labelOnlyBase : bool, optional, default: False
        If True, label ticks only at integer powers of base.
        This is normally True for major ticks and False for
        minor ticks.
    minor_thresholds : (subset, all), optional, default: (1, 0.4)
        If labelOnlyBase is False, these two numbers control
        the labeling of ticks that are not at integer powers of
        base; normally these are the minor ticks. The controlling
        parameter is the log of the axis data range.  In the typical
        case where base is 10 it is the number of decades spanned
        by the axis, so we can call it 'numdec'. If ``numdec <= all``,
        all minor ticks will be labeled.  If ``all < numdec <= subset``,
        then only a subset of minor ticks will be labeled, so as to
        avoid crowding. If ``numdec > subset`` then no minor ticks will
        be labeled.
    linthresh : None or float, optional, default: None
        If a symmetric log scale is in use, its ``linthresh``
        parameter must be supplied here.
        
    **Notes**

    The `set_locs` method must be called to enable the subsetting
    logic controlled by the ``minor_thresholds`` parameter.
    In some cases such as the colorbar, there is no distinction between
    major and minor ticks; the tick locations might be set manually,
    or by a locator that puts ticks at integer powers of base and
    at intermediate locations.  For this situation, disable the
    minor_thresholds logic by using ``minor_thresholds=(np.inf, np.inf)``,
    so that all ticks will be labeled.
    To disable labeling of minor ticks when 'labelOnlyBase' is False,
    use ``minor_thresholds=(0, 0)``.  This is the default for the
    "classic" style.
    
    **Examples**

    To label a subset of minor ticks when the view limits span up
    to 2 decades, and all of the ticks when zoomed in to 0.5 decades
    or less, use ``minor_thresholds=(2, 0.5)``.
    To label all minor ticks when the view limits span up to 1.5
    decades, use ``minor_thresholds=(1.5, 1.5)``.
    
    """

    def __init__(self, labelOnlyBase=True,
                 minor_thresholds=None,
                 linCutOff=20):

        self.labelOnlyBase = labelOnlyBase
        if minor_thresholds is None:
            if rcParams['_internal.classic_mode']:
                minor_thresholds = (0, 0)
            else:
                minor_thresholds = (1, 0.4)
        self.minor_thresholds = minor_thresholds
        self._sublabels = None
        self._linthresh = linCutOff
        self._base = np.exp(1)

    def _num_to_string(self, x, vmin, vmax):
        x = round(x, 0)
        s = self.pprint_val(x, vmax - vmin)
        return s

    def __call__(self, x, pos=None):
        """
        Return the format for tick val `x`.
        """
        if x == 0.0:  # Symlog
            return '0'
        vmin, vmax = self.axis.get_view_interval()

        tx = invertBiLogPlotcoordinate(x, vmin, vmax, self._linthresh)
        tx = np.round(tx)

        if tx > self._linthresh + 1:
            fx = math.log(tx) / math.log(10.0)
            is_x_decade = is_close_to_int(fx)
            exponent = np.round(fx) if is_x_decade else np.floor(fx)
            coeff = np.round(x / 10.0 ** exponent)

            if self.labelOnlyBase and not is_x_decade:
                return ''
            if self._sublabels is not None and coeff not in self._sublabels:
                return ''
        elif tx < -self._linthresh - 1:
            fx = math.log(abs(tx)) / math.log(10.0)
            is_x_decade = is_close_to_int(fx)
            exponent = np.round(fx) if is_x_decade else np.floor(fx)
            coeff = -np.round(x / 10.0 ** exponent)

            if self.labelOnlyBase and not is_x_decade:
                return ''
            if self._sublabels is not None and coeff not in self._sublabels:
                return ''
        else:
            # Manually define acceptable negative values
            accepted_range = list([-self._linthresh, 100, 0, 100, self._linthresh])
            if not np.round(tx) in accepted_range:
                return ''
        s = self._num_to_string(tx, vmin, vmax)
        return self.fix_minus(s)

    def pprint_val(self, x, d):
        # If the number is at or below the set lin-cutoff (_lintrehsh)
        # Print it as an int

        if x <= self._linthresh + 1 and x >= -self._linthresh - 1:
            return '%d' % x

        fmt = '%1.3e'
        s = fmt % x
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            if exponent:
                if float(mantissa) > 1:
                    s = '$%s*10^{%d}$' % (mantissa, exponent)
                else:
                    s = '$%s0^{%d}$' % (mantissa, exponent)
            else:
                s = mantissa
        else:
            s = s.rstrip('0').rstrip('.')
        return s


class SymmetricalLogLocator(Locator):
    # Modified from matplotlibs SymmetricalLogLocator
    # https://matplotlib.org/3.1.1/_modules/matplotlib/ticker.html#SymmetricalLogLocator
    """
    Determine the tick locations for symmetric log axes
    """

    def __init__(self, subs=None, linthresh=None):
        """
        place ticks on the location= base**i*subs[j]
        """
        self._base = np.exp(1)
        self._linthresh = linthresh

        if subs is None:
            self._subs = [1.0]
        else:
            self._subs = subs
        self.numticks = 15

    def set_params(self, subs=None, numticks=None):
        """Set parameters within this locator."""
        if numticks is not None:
            self.numticks = numticks
        if subs is not None:
            self._subs = subs

    def __call__(self):
        'Return the locations of the ticks'
        # Note, these are untransformed coordinates
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        b = self._base
        t = self._linthresh

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # The domain is divided into three sections, only some of
        # which may actually be present.
        #
        # <======== -t == 0 == t ========>
        # aaaaaaaaa    bbbbb   ccccccccc
        #
        # a) and c) will have ticks at integral log positions.  The
        # number of ticks needs to be reduced if there are more
        # than self.numticks of them.
        #
        # b) has a tick at 0 and only 0 (we assume t is a small
        # number, and the linear segment is just an implementation
        # detail and not interesting.)
        #
        # We could also add ticks at t, but that seems to usually be
        # uninteresting.
        #
        # "simple" mode is when the range falls entirely within (-t,
        # t) -- it should just display (vmin, 0, vmax)

        has_a = has_b = has_c = False
        if vmin < -t:
            has_a = True
            if vmax > -t:
                has_b = True
                if vmax > t:
                    has_c = True
        elif vmin < 0:
            if vmax > 0:
                has_b = True
                if vmax > t:
                    has_c = True
            else:
                return [vmin, vmax]
        elif vmin < t:
            if vmax > t:
                has_b = True
                has_c = True
            else:
                return [vmin, vmax]
        else:
            has_c = True

        def get_log_range(lo, hi):
            lo = np.floor(np.log(lo) / np.log(b))
            hi = np.ceil(np.log(hi) / np.log(b))
            return lo, hi

        # First, calculate all the ranges, so we can determine striding
        if has_a:
            if has_b:
                a_range = get_log_range(t, -vmin + 1)
            else:
                a_range = get_log_range(-vmax, -vmin + 1)
        else:
            a_range = (0, 0)

        if has_c:
            if has_b:
                c_range = get_log_range(t, vmax + 1)
            else:
                c_range = get_log_range(vmin, vmax + 1)
        else:
            c_range = (0, 0)

        total_ticks = (a_range[1] - a_range[0]) + (c_range[1] - c_range[0])
        if has_b:
            total_ticks += 1
        stride = max(total_ticks // (self.numticks - 1), 1)

        decades = []
        if has_a:
            decades.extend(-1 * (b ** (np.arange(a_range[0], a_range[1],
                                                 stride)[::-1])))

        if has_b:
            decades.append(0.0)

        if has_c:
            decades.extend(b ** (np.arange(c_range[0], c_range[1], stride)))

        # Add the subticks if requested
        if self._subs is None:
            subs = np.arange(2.0, b)
        else:
            subs = np.asarray(self._subs)

        if len(subs) > 1 or subs[0] != 1.0:
            ticklocs = []
            for decade in decades:
                if decade == 0:
                    ticklocs.append(decade)
                else:
                    ticklocs.extend(subs * decade)
        else:
            ticklocs = decades

        return self.raise_if_exceeds(np.array(ticklocs))

    def view_limits(self, vmin, vmax):
        'Try to choose the view limits intelligently'
        b = self._base
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if rcParams['axes.autolimit_mode'] == 'round_numbers':
            if not is_decade(abs(vmin), b):
                if vmin < 0:
                    vmin = -decade_up(-vmin, b)
                else:
                    vmin = decade_down(vmin, b)
            if not is_decade(abs(vmax), b):
                if vmax < 0:
                    vmax = -decade_down(-vmax, b)
                else:
                    vmax = decade_up(vmax, b)

            if vmin == vmax:
                if vmin < 0:
                    vmin = -decade_up(-vmin, b)
                    vmax = -decade_down(-vmax, b)
                else:
                    vmin = decade_down(vmin, b)
                    vmax = decade_up(vmax, b)

        result = mtransforms.nonsingular(vmin, vmax)
        return result
