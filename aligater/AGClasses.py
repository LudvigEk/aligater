#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#	______|\__________\o/__________
#			~~aliGater~~
#	(semi)automated gating software
#
#               /^^\
#   /^^\_______/0  \_
#  (                 `~+++,,_________,,++~^^^^^^^
#..V^V^V^V^V^V^\.................................
#
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	http://nilssonlab.org

import numpy as np
from matplotlib.ticker import Locator, Formatter
import aligater as ag
from matplotlib import rcParams
import pandas as pd
import sys, os, errno
import six

sentinel=object()

class AGgate(object):
    current=[]
    parent=[]
    
    def __init__(self, vI, vIParent):
        self.current=vI
        self.parent=vIParent
    
    def __call__(self):
        if not list(self.current):
            sys.stderr.write("This AGGate object does not contain any data\n")
            return []
        return self.current
    
    def changeParent(self, vIParent):
        self.parent=vIParent
        #THIS A GOOD IDEA?
        return self
    
    def getParent(self):
        #if not self.parent:
            #raise ValueError("This AGGate object does not contain any parent")
        return self.parent
    
    def reportStats(self):
        if not list(self.current):
            raise ValueError("This AGGate object does not contain any data")
        elif not list(self.parent):
            sys.stderr.write("This AGGate object does not have any parent, only reporting current\n")
            return str(len(self.current))
        else:
            nOfcurrent = float(len(self.current))
            nOfparent = float(len(self.parent))
            if any([nOfcurrent==0, nOfparent==0]):
                sys.stderr.write("this AGGate object have reached 0 population (no gated events)\n")
                return str(nOfparent)+"\t"+str(nOfcurrent)+"\t0"
            outputStr=str(nOfparent)+"\t"+str(nOfcurrent)+"\t"+str(nOfcurrent/nOfparent)
            return outputStr
    

class AGsample(object):
    #vGate intended as list of tuple(name, gate)
    vGates=None
    sample=None
    filePath=None
    fcsDF=None
    gate=None
    
    def __init__(self, fcsDF, filePath, sampleName = sentinel):
        if not isinstance(fcsDF, pd.DataFrame):
            raise
        if not isinstance(filePath, str):
            raise
        if sampleName is not sentinel:
            if not isinstance(sampleName, str):
                raise
        else:
            #If no name specified, set it to the filename without extension
            sampleName=ag.getFileName(filePath)
            
        self.fcsDF=fcsDF
        self.sample=sampleName
        self.vGates=[]
        self.filePath=filePath
    
    def __call__(self, name=sentinel):
        if name is sentinel:
            if not isinstance(self.fcsDF, pd.DataFrame):
                sys.stderr.write("sample does not contain any data\n")
                return None
            return self.fcsDF
        else:
            if not isinstance(name, str):
                raise
            else:
                if not self.vGates:
                    sys.stderr.write("sample does not contain any gates\n")
                    return None
                for index, elem in enumerate(self.vGates):
                    if elem[0]==name:
                        return elem[1]()
                sys.stderr.write(name+" not in sample name list\n")
                return None
        
    def update(self, gate, name, parentName):
        if not isinstance(gate,AGgate):
            raise TypeError("gate is not a valid AGgate object")
        if not isinstance(name, str):
            raise TypeError("name must be specified as a string")
        if not isinstance(parentName, str):
            raise TypeError("parent name must be specified as a string")
        self.vGates.append((name, gate, parentName))
    
    def full_index(self):
        return list(self.fcsDF.index.values)
    
    def report(self):
        reportStr="Gating report for "+self.sample+" ("+str(len(self.fcsDF.index))+" total events)\n"+str(len(self.vGates))+" gate(s):\n"
        for gate in self.vGates:
            reportStr=reportStr+str(gate[0])+" ("+str(len(gate[1]()))+" events)\n"
        print(reportStr)
    
    def printData(self, file, precision=4, header=True):
        if header:
            #header
            header="date\tplate\tSample\tTotal"
            for gate in self.vGates:
                header=header+"\t"+str(gate[0])+"\t"+str(gate[0])+"/"+str(gate[2])
            header=header+"\n"
            file.write(header)
        #Data
        reportStr=ag.getFileName(ag.getParent(ag.getParent(self.filePath)))+"\t"+ag.getFileName(ag.getParent(self.filePath))+"\t"+self.sample+"\t"+str(len(self.full_index()))
        precision="."+str(precision)+"f"
        for gate in self.vGates:
            if len(gate[1].getParent()) == 0 or len(gate[1]()) == 0:
                fraction=str(0)
            else:
                fraction=str(format(len(gate[1]())/len(gate[1].getParent()),precision))
            reportStr=reportStr+"\t"+str(len(gate[1]()))+"\t"+fraction
        reportStr=reportStr+"\n"
        file.write(reportStr)
        return None
    
    def printStats(self, output=None, overwrite=False, precision=4, header=True):
        if isinstance(output,str):
            output=output
            if os.path.isfile(output):
                if overwrite:
                    file = open(output, 'w')
                    self.printData(output, precision, header)
                else:
                    file = open(output, 'a')
                    self.printData(file, precision, header)
            else:
                try:
                    os.makedirs(os.path.dirname(output))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
                if header:
                    file = open(output, 'w')
                else: 
                    file = open(output, 'a')
                self.printData(file, precision, header)
    
class AGExperiment(object):
    fcsList=[]              #LIST
    normaliseLevel=None     #STR
    nOfPlates=0             #INT    
    normalise=None          #BOOL
    plateList=[]            #LIST
    
    def __init__(self, experimentRoot, *args, **kwargs):
        #optional arguments for ag.collectFiles call
        lFilter=None
        lMask=None
        lIgnoreTypes=None
        if 'filter' in kwargs:
            if not isinstance(kwargs['filter'],str):
                if not isinstance(kwargs['filter'],list):
                    raise TypeError("if filter is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['filter']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in filter")
                    lFilter=kwargs['filter']
            else:
                lFilter=[kwargs['filter']]
            reportStr=str(len(lFilter))+" filter(s) defined\n"
            sys.stderr.write(reportStr)
            
        if 'mask' in kwargs:
            if not isinstance(kwargs['mask'],str):
                if not isinstance(kwargs['mask'],list):
                    raise TypeError("if mask is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['mask']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in mask")
                    lMask=kwargs['mask']
            else:
                lMask=[kwargs['mask']]
            reportStr=str(len(lMask))+" mask(s) defined\n"
            sys.stderr.write(reportStr)
            
        if 'ignoretype' in kwargs:
            if not isinstance(kwargs['ignoretype'],str):
                if not isinstance(kwargs['ignoretype'],list):
                    raise TypeError("if ignoretype is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['ignoretype']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in mask")
                    lIgnoreTypes=kwargs['ignoretype']
            else:
                lIgnoreTypes=[kwargs['ignoretype']]
            reportStr=str(len(lIgnoreTypes))+" ignoretype defined\n"
            sys.stderr.write(reportStr)
            
        if 'normaliseOn' in kwargs:
            if not isinstance(kwargs['normaliseOn'],str):
                raise (TypeError("normaliseOn must be string"))
            self.normalise=True
            self.normaliseLevel=kwargs['normaliseOn']
            reportStr="Plate normalisation/outlier detection requested\n"
            sys.stderr.write(reportStr)
        

        #If no normalisation, collect all files
        self.fcsList=ag.collectFiles(experimentRoot,lFilter,lMask,lIgnoreTypes)
        if self.normalise:
            #Otherwise (plate normalisation) collect abs path to all end-of-tree folders
            self.plateList=ag.listDir(experimentRoot)
            self.checkPlateList(lFilter,lMask,lIgnoreTypes)
            reportStr="Files are distributed in "+str(len(self.plateList))+" plates\n"
            sys.stderr.write(reportStr)
    
    def checkPlateList(self,lFilter,lMask,lIgnoreTypes):
        #Function that goes through the list of folders and removes those that do not contain fcs files
        #The remaining folders are considered plates
        lFlaggedFolders=[]
        for index,folder in enumerate(self.plateList):
            filesInFolder=next(os.walk(folder))[2]
            if not filesInFolder:
                lFlaggedFolders.append(index)
        lFlaggedFolders=np.sort(lFlaggedFolders)[::-1]
        for index in lFlaggedFolders:
            del self.plateList[index]
        
    def organize(self,*args,**kwargs):
        
        return True

         
def is_decade(x, base=10):
    if not np.isfinite(x):
        return False
    if x == 0.0:
        return True
    lx = np.log(np.abs(x)) / np.log(base)
    return is_close_to_int(lx)

def is_close_to_int(x):
    if not np.isfinite(x):
        return False
    return abs(x - nearest_int(x)) < 1e-10

def nearest_int(x):
    if x == 0:
        return int(0)
    elif x > 0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)

        
def convertToLogishPlotCoordinates(Ticlocs, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = ag.logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    tTiclocs=ag.logishTransform(Ticlocs, T)
    plotTics=[]
    for tTic in tTiclocs:
        plotTic=(tTic-tMinMax[0])/transformedRange*actualRange+vmin
        plotTics.append(plotTic)
    assert len(tTiclocs)==len(Ticlocs)
    return plotTics

def convertToLogishPlotCoordinate(Ticloc, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = ag.logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    tTicloc=ag.logishTransform([Ticloc], T)[0]
    plotTic=(tTicloc-tMinMax[0])/transformedRange*actualRange+vmin
    return plotTic

def invertLogishPlotcoordinates(plotTics, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = ag.logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    invPlotTics=[]
    for tTic in plotTics:
        invPlotTic=(tTic-vmin)/actualRange*transformedRange+tMinMax[0]
        invPlotTics.append(invPlotTic)
    result=ag.inverseLogishTransform(invPlotTics, T)
    return result

def invertLogishPlotcoordinate(plotTic, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = ag.logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    invPlotTic=(plotTic-vmin)/actualRange*transformedRange+tMinMax[0]
    result=ag.inverseLogishTransform([invPlotTic], T)[0]
    return result

class LogishLocator(Locator):
    """
    Determine the tick locations for logish axes based on LogLocator
    Hacked version of LogLogator that covers normal usecases of the logish scale
    Only defined with ticlocations for data in range -50000 < x < 1 000 000
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
                
        self._base=np.exp(1)
        self.subs(subs)
        self.numdecs = numdecs
        self.numticks = numticks
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
        
    #if vmin < self.T:
        tmpTicLoc=[-50000, -40000, -30000, -20000, -10000, -5000,-4000,-3000, -2000, -1000, 0]
        Ticlocs = list(set(np.clip(tmpTicLoc, vmin, self.T)))
        Ticlocs = list(np.sort(Ticlocs))
    
    #if vmax > self.T:
        tmpTicLoc=[0,1000]
        tmpTicLoc.extend(np.arange(1000.0,10001,1000))
        tmpTicLoc.extend(np.arange(10000.0,100001,10000))
        tmpTicLoc.extend(np.arange(100000.0,1000000,100000)) #[10000.0,100000.0, 200000, 300000, 1000000.0]
        Ticlocs.extend(tmpTicLoc)
        clip_Ticlocs=list(set(np.clip(Ticlocs,vmin, vmax)))
        Ticlocs=np.sort(clip_Ticlocs)
        #ADD HOC POSSIBLY
        Ticlocs=Ticlocs[1:-1]
        Ticlocs=convertToLogishPlotCoordinates(Ticlocs, vmin, vmax, self.T)
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        return self.raise_if_exceeds(np.asarray(Ticlocs))

    def view_limits(self, vmin=None, vmax=None):
        'Try to choose the view limits intelligently'
        vmin, vmax = self.axis.get_view_interval()
        return vmin, vmax


class LogishFormatter(Formatter):
    """
    Base class for formatting ticks on a logish scale.
    Hacked version of LogFormatter that covers normal usecases of the logish scale
    Only defined with formatting ticlabels for data in range -50000 < x < 1 000 000

    
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
    def __init__(self, labelOnlyBase=False,
                 minor_thresholds=None,
                 linthresh=1000):
        
        self.labelOnlyBase = labelOnlyBase
        if minor_thresholds is None:
            if rcParams['_internal.classic_mode']:
                minor_thresholds = (0, 0)
            else:
                minor_thresholds = (1, 0.4)
        self.minor_thresholds = minor_thresholds
        self._sublabels = None
        self._linthresh = linthresh
        self._base = np.exp(1)



    def _num_to_string(self, x, vmin, vmax):
        x = round(x,0)
        if not x in [-5000, -4000, -3000, -2000, -1000, 0 ,1000,10000,100000,1000000]:
            s = ''
        else:
            s = self.pprint_val(x, vmax - vmin)
        return s

    def __call__(self, x, pos=None):
        """
        Return the format for tick val `x`.
        """
        if x == 0.0:  # Symlog
            return '0'
        vmin, vmax = self.axis.get_view_interval()
        #tVals = ag.logishTransform([vmin, vmax, x], self._linthresh)
        # only label the decades
        #fx = (x-vmin)/(vmax-vmin)*(tVals[1] - tVals[0])-tVals[0]
        #fx = ag.inverseLogishTransform([fx],self._linthresh)[0]
        fx=invertLogishPlotcoordinate(x,vmin,vmax,self._linthresh)
        #print(fx)
        s = self._num_to_string(fx, vmin, vmax)
        return self.fix_minus(s)
        

    def pprint_val(self, x, d):
        #If the number is at or below the set lin-cutoff (_lintrehsh)
        #Print it as an int
        #TODO: WHY DO I NEED THE +1 HERE?
        if x <= self._linthresh+1:
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