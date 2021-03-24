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
import pandas as pd
import sys, os, errno
import datetime #For creating output folder names

#These are for i/o operations used by the AGQC object to store downsampled images
import tempfile
from shutil import copyfile
import h5py
from matplotlib.pyplot import close as close_pyplot_figures

#AliGater imports
import aligater.AGConfig as agconf
from aligater.check_exec_mode import check_filePath
from aligater.AGPlotRoutines import getHeatmap, imagePCA_cluster, plot_flattened_heatmap
from aligater.AGFileSystem import getGatedVector, getGatedVectors, getFileName, getParent, collectFiles, listDir, loadFCS, getCompensationMatrix, AliGaterError, invalidAGgateError, check_exists, applyFilter, loadHDF5sample
#API to the FlowCytometryTools parser
from aligater.fscparser_api import parse

sentinel=object()   


def update_progress(progress):
#Imported StackOverflow progressbar method. Credit to user Brian Khuu.
#https://stackoverflow.com/questions/3160699/python-progress-bar/15860757#15860757
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), format(progress*100,'.2f'), status)
    sys.stderr.write(text)
    sys.stderr.flush()


def gateIndexWrapper(func, gateObj):
    #Deprecated
    if gateObj is None:
        return None
    if not isinstance(gateObj, AGgate):
        reportStr='Invalid AGgate obj encountered in: '+func+", expected AGClasses.AGgate, found "+str(type(gateObj))+"\n"
        raise TypeError(reportStr)
    else:
        return gateObj.current()

    
class AGgate:
    """
    **Overview**

    | Object for containing results from a gating operation.
    | In essence it contains information on which events belong to a gated population and the name of it's parent population.
    |  
    | Optionally it can hold additional information such as which markers labels where on the x- and y-axis', and a downsampled version of the pre-gate view.
    |  
    | Such additional information is useful for automating formatted output from a batch gating experiment.
    |  
    
    **Members**
    
    current, list-like
        List-like index tracking rows(events) correspond to a gated population in a AGClasses.AGSample.
    parent, list-like, optional, default: None
        | List-like index tracking rows(events) correspond to a population's parent population in a AGClasses.AGSample.
        | If not passed, assumes that the parent population is the ungated sample.
    xCol, str, optional, default: None
        | User-input string name of the x-axis marker label. 
        | NOTE: Will **not** raise on non-str input, instead sets to None.
    yCol, str, optional, default: None
        | User-input string name of the y-axis marker label. 
        | NOTE: Will **not** raise on non-str input, instead sets to None.
    name, str
        | User-input string name of the gated population.
        | Raises an exception on non-str input.
    parentName, str or None, default: None
        User-input string name of the gated population's parent population.
    RatioGate, bool, optional, default: False
        If True, only report the percent-of-parent population for this gate (i.e. the ratio between the two input populations)\n
        It will also ignore the checks to the minCells limit.
    IgnoreCellLimit, bool, optional, default: False
        If True, will bypass the check to the minCells limit\n
        Useful for late gates that are known to have small number events but still should be gated, perhaps with fix unflexible gates.\n
        This makes it harder to quality control larger sets of samples for this gate, use carefully.
    **Internal members**

    | These are used by other, internal, AliGater functions and are not ment to be interacted with.
    |  
    
    m_downSample
        Container for a downsampled heatmap of the pre-gating view.
    bInvalid
        | Invalid flag, used for differentiating between a depleted population (0-events) that should be considered and an invalid population that should be disregarded.
        |  
        | NOTE: Will be set to True in any gate objects if number of events in parentGate is inclusive less than the threshold agconf.minCells. 
        | See examples.
    bNoparent
        Flag for identifying samples with no parent. If true, these gate objects are assumed to stem from the ungated full sample.
    
    **Methods**
    
    __init__(self, gate, parentGate, xCol, yCol, name)
        Initialization method, see examples.
    __call__(self)
        | Function for access to the current member with checks for flags and no data.
        | Should be preferred over direct access of the current member.
    changeParent(self, parent, name)
        parent, AGClasses.AGgate
            Parent population, expects another AGClasses.AGgate object.
        name, str, optional, default: None
            Name of parent population.
    getParent(self)
        | Function for access to the parent member with checks for flags and no data.
        | Similar to the __call__ function this is the preferred method for access to the parent member.
    reportStats(self)
        | Function that prints a formatted report of the AGgate objects content. 
        |  
        | Useful when exploring data, but is not ment to be used for printing gating results in larger batch experiments.
        | For reporting gating results of an experiment see AGClasses.AGExperiment and it's printExperiment function.
    
    **Internal Methods**
    
    These are used by other, internal, AliGater functions and are not ment to be interacted with.
    
    downSample
        Function that creates and stores a downsampled version of the gating view.
        
    setInvalid
        Sets the bInvalid member to True
    
    **Examples**

    None currently.
    """
    current=[]
    parent=[]
    xCol=None
    yCol=None
    name=None
    parentName=None
    m_downSample=None
    bInvalid=False
    bNoparent=False
    bQC=False
    bRatioGate=False
    bIgnoreCellLimit=False
    
    def __init__(self, gate, parentGate, xCol, yCol, name=None, RatioGate=False, IgnoreCellLimit=False):
        if not isinstance(gate, AGgate):
            if not isinstance(gate,list):
                raise AliGaterError("Invalid init of AGClasses.AGgate object.")
            else:
                self.current=gate
                if name is None or not isinstance(name, str):
                    raise AliGaterError("Invalid init of AGClasses.AGgate object. If gate argument is list-like, a valid name has to be provided.")
                self.name=name
        else:
            self.current=gate()
            if name is None or not isinstance(name, str):
                if gate.name is None or not isinstance(gate.name, str) :
                    raise AliGaterError("Invalid init of AGClasses.AGgate object, passed gate object has an invalid name and no overwriting name was passed.")
                else:
                    self.name=gate.name
            else:
                self.name = name

            
        if not isinstance(RatioGate,bool):
            raise TypeError("unexpected type of bRatioGate. Expected bool, found: "+str(type(RatioGate)))
        self.bRatioGate=RatioGate
        if not isinstance(IgnoreCellLimit,bool):
            raise TypeError("unexpected type of IgnoreCellLimit. Expected bool, found: "+str(type(IgnoreCellLimit)))
        self.bIgnoreCellLimit=IgnoreCellLimit
        # if name is None or not isinstance(name, str):
        #     raise ValueError("No, or invalid, name specified for gate")
        # self.name=name
        if not isinstance(parentGate, AGgate):
            if parentGate is not None:
                raise AliGaterError("Invalid init of AGClasses.AGgate object, parentGate must either be AGClasses.AGgate or None.")
            else:
                self.parent=None
                self.parentName="total"    
                self.bNoparent=True
        else:
            self.parent=parentGate()
            self.parentName=parentGate.name
        if not isinstance(xCol, str):
            self.xCol=None
        else:
            self.xCol=xCol
        if not isinstance(yCol, str):
            self.yCol=None
        else:
            self.yCol=yCol
        #Flag gate if parent has fewer cells than minCells    
        if not self.bRatioGate and not self.bNoparent and not self.bIgnoreCellLimit:
            if (len(self.parent) < agconf.minCells):
                self.bInvalid=True
        
    def __call__(self):
        if self.bInvalid:
            #Invalid flag
            sys.stderr.write("WARNING: Call to AGGate object ("+self.name+") that has had its invalid flag set\n")
        if not isinstance(self.current, list):
            #Invalid dtype
            raise AliGaterError("This AGGate object ("+self.name+") had a non-list current member\n")
        if not list(self.current):
            #Empty
            return []
        return self.current
    
    def changeParent(self, parent=sentinel, name=None):
        if parent is sentinel:
            self.parent=None
            self.parentName="total"
            self.bNoparent=True
            return
        if not isinstance(parent, AGgate):
            raise TypeError("Error in changeParent call: Passed parent population to AGgate object "+self.name+" is not a AGClasses.AGgate object")
        self.parent=parent()
        self.parentName=parent.name
        self.bNoparent=False
        return
    
    def getParent(self):
        if not list(self.parent):
            reportStr="This AGGate object does not contain any parent data\nIf you want the full index, access it through the AGClasses.AGgate object"
            raise ValueError(reportStr) 
        return self.parent
    
    def downSample(self, fcsDF, bins, xlim, ylim, scale='linear', xscale='linear', yscale='linear', T=1000):
        vX, vY = getGatedVectors(fcsDF, self.xCol, self.yCol, vI=self.parent, return_type="nparray")
        #flaggedIndicies=[]
        for i in np.arange(0,len(vX),1):
            if vX[i]<xlim[0]:
                vX[i]=xlim[0]
            if vX[i]>xlim[1]:
                vX[i]=xlim[1]
            if vY[i]<ylim[0]:
                vY[i]=ylim[0]
            if vY[i]>ylim[1]:
                vY[i]=ylim[1]
        heatmapRange=[xlim, ylim]
        #vX=np.delete(vX, flaggedIndicies)
        #vY=np.delete(vY, flaggedIndicies)
        self.m_downSample = getHeatmap(vX, vY, bins, scale, xscale, yscale, T=T, normalize=True, range=heatmapRange)[0]
        self.bQC=True
        
    def setInvalid(self):
        self.bInvalid=True
    
    def reportStats(self):
        if not list(self.current):
            raise AliGaterError("This AGGate object ("+self.name+") does not contain any data")
        elif not list(self.parent):
            sys.stderr.write("This AGGate object ("+self.name+") does not have any parent, only reporting current\n")
            return str(len(self.current))
        else:
            nOfcurrent = float(len(self.current))
            nOfparent = float(len(self.parent))
            if any([nOfcurrent==0, nOfparent==0]):
                sys.stderr.write("This AGGate object ("+self.name+") have reached 0 population (no gated events)\n")
                return str(nOfparent)+"\t"+str(nOfcurrent)+"\t0"
            outputStr=str(nOfparent)+"\t"+str(nOfcurrent)+"\t"+str(nOfcurrent/nOfparent)
            return outputStr
    
    

class AGsample:
    """
    **Overview**

    The AGSample object contains the data-matrix from the .fcs file input. It also keeps a list of all the AGgate objects that has been applied to it.

    Any user-made strategy should accept an AGSample as input, apply gate functions to it and then return the modified AGSample back to AliGater

    The Pandas DataFrame (the fcsDF member) should never be interacted with directly by the user. If python supported const, it would be const.
    See notes for more on this.
      
    **Members**
    
    sample, str, optional, default: See below
    
        User-given sample name.
        
        | Notes on default behaviour:
        | If no explicit sample name is given, the name will default to the filename plus the name of its two parent folders.
        | I.e. if a .fcs file is located in /C/project_name/fcs_files/plate_3/my_fcs_file.fcs it would get the name 'fcs_files/plate_3/my_fcs_file.fcs'
        | Settings to tweak this default behaviour will be implemented.
        
    **Internal members**

    vGates
        List-like of AGgate objects that belong to this sample.

    filePath
        Absolute str path to the fcs file.
        
        NOTE: if .fcs file paths are changed during runtime, an error will be raised.

    fcsDF
        The loaded fcs data stored as a pandas DataFrame.

    **Methods**
    
    __call__(name)
        This function can be used to get the full ungated dataframe if no name is given, otherwise can return a specific earlier gate.
        
        See sample notebooks for uses.
    
    update(gate, QC=False)
      gate, AGClasses.AGgate
          A gate object that should be added to the samples gate list.
      QC, bool, default: False
          A flag deciding if a downsampled picture of the gating view should be saved.
          Should be set to True if: 

          A) You want QC post-analysis through PCA and inspection of the downsampled gating views. 

          And
    
          B) Not for intermediate gating steps (these will be worthless in your post-analysis QC).
          
          See example notebook on QC for best practice
    
    report()
        Function that reports a quick summary of this saples total events and the gates that have been applied to it.

    **Internal Methods**
    
    These are used by other, internal, AliGater functions and are not ment to be interacted with.
    
    __init__(fcsDF, filePath, sampleName)
        Initialization method. Intended to either be called by an AGExperiment object or through the AGFileSystem.loadFCS function.
        
    printData(file, precision=4, header=True)
        Function for printing formatted content to a filehandle.
    
    printStats(output=None, overwrite=False, precision=4, header=True)    
        Wrapper function for printData that does file checking, and sets options like overwriting vs appending.
        
    resultVector(order=None, precision='.3f')
        Function that creates a list- or matrix-like table of reported ratios and events in this sample.
        Called by an AGExperiment object.
    
    **Examples**
    
    None currently.
    
    **Notes**
    
    On the fcsDF member:
     
    To ensure that no changes has been made to the fcsDF, inadvertently or intentionally, it would be possible for AliGater to store a copy and make a comparison with that copy after each user-strategy has been applied.
    
    This can have a significant performance impact depending on how many events have been recorded in each sample and has **thus been left out by design**.

    """
    #vGate intended as list of tuple(name, gate)
    vGates=None
    sample=None
    filePath=None
    fcsDF=None
    _downsamplingBins=None
    MFI_Series=pd.Series()
    
    def __init__(self, fcsDF, filePath, sampleName = sentinel, sampling_resolution=32):
        if not isinstance(fcsDF, pd.DataFrame):
            raise
        if not isinstance(filePath, str):
            raise
        if not isinstance(sampling_resolution, int):
            raise
        if sampleName is not sentinel:
            if not isinstance(sampleName, str):
                raise
        else:
            #If no name specified, set it to the filename without extension
            datename=getFileName(getParent(getParent(filePath)))
            platename=getFileName(getParent(filePath))
            sampleName=datename+"/"+platename+"/"+getFileName(filePath)
            
        self.fcsDF=fcsDF
        self.sample=sampleName
        self.MFI_Series.name = sampleName
        self.vGates=[]
        self.filePath=filePath
        self._downsamplingBins=sampling_resolution
    
    def __call__(self, name=sentinel):
        if name is sentinel:
            if not isinstance(self.fcsDF, pd.DataFrame):
                sys.stderr.write("sample does not contain any data\n")
                return None
            return self.fcsDF
        else:
            if not isinstance(name, str):
                raise AliGaterError("TypeError in AGSample call,","name argument must be string")
            else:
                if not self.vGates:
                    sys.stderr.write("sample does not contain any gates\n")
                    return None
                for index, gate in enumerate(self.vGates):
                    if gate.name==name:
                        return gate
                reportStr=name+" not in sample name list\n"
                sys.stderr.write(reportStr)
                return False
        
    def update(self, gate, xlim=[0,500000], ylim=[0,500000], QC=False, scale='linear', xscale='linear', yscale='linear', T=1000, MFI=False, MFI_type="current", extra_MFI=None):
        if not isinstance(gate,AGgate):
            raise invalidAGgateError("in AGsample.update: ")
        if gate.xCol not in self.fcsDF.columns:
            raise AliGaterError("in update: ","x marker label ("+str(gate.xCol)+") doesn't exist in the sample")
        if gate.xCol not in self.fcsDF.columns:
            raise AliGaterError("in update: ","y marker label ("+str(gate.yCol)+") doesn't exist in the sample")

        #*************************************************
        validName=False
        nameChange=False
        newNameCounter=0
        if self.vGates is not None: #Only check for existing gates if there are any
            new_name=gate.name #initialize to current name
            while not validName:
                for existing_gate in self.vGates:
                    if existing_gate.name == new_name:
                        #A gate with the same name already exists
                        newNameCounter+=1
                        nameChange=True #Flag that name has to be changed
                        new_name = gate.name+str(newNameCounter)
                        break
                else:
                    validName=True
                    
            if nameChange: #Update name is needed
                gate.name = new_name

        #*****************MFI STUFF*****************************************************
        if MFI:
            if MFI_type is not None:
                if isinstance(MFI_type, str):
                    if not any([MFI_type.lower() in ['current', 'all']]): #Option is to just remove the all option
                        raise AliGaterError("in update: ","MFI_type must be None or string: 'current' or 'all'")
                    if MFI_type.lower() == 'current':
                        self.collect_current_MFI(gate=gate)                        
                    elif MFI_type.lower() == 'all':
                        self.collect_all_MFI(gate=gate) #This probably doesn't work atm (after MFI fix reversal)
                else:
                    raise AliGaterError("in update: ","MFI_type must be None or string: 'current' or 'all'")

            if extra_MFI is not None:
                if not isinstance(extra_MFI,(list,np.ndarray)):
                    raise AliGaterError("in update: ","extra_MFI must be None or a list/array of string labels of MFIs to save for current population.")
                if len(extra_MFI)>=1:
                    self.collect_extra_MFI(gate = gate, MFIs_to_collect = extra_MFI)
        #******************************************************************************
        self.vGates.append(gate)
        if QC:
            if not gate.bInvalid:                
                gate.downSample(self.fcsDF, self._downsamplingBins, xlim, ylim, scale, xscale, yscale, T=T)
    
    def collect_current_MFI(self, gate):
        if gate.xCol is not None:
            self.collect_MFI(gate=gate, marker=gate.xCol)
        if gate.yCol is not None:
            self.collect_MFI(gate=gate, marker=gate.yCol)
        return None
    
    def collect_MFI(self, gate, marker):
        if "FSC" in marker or "SSC" in marker:
            return None
        #Collect MFI of a single marker for an AGGate object
        vX = getGatedVector(self.fcsDF, gate = marker, vI=gate(), return_type='nparray')
        x_mean = np.mean(vX)
        x_mean_MFI_label = gate.name +"_" + marker + "_mean"
        
        x_median = np.median(vX)
        x_median_MFI_label = gate.name +"_" + marker + "_median"
        
        self.MFI_Series = self.MFI_Series.append(pd.Series(data=[x_mean, x_median], index=[x_mean_MFI_label,  x_median_MFI_label], name=self.sample))
        return None

    def collect_all_MFI(self, gate):
        #Just sanity check for empty list in self
        if self.vGates is None or len(self.vGates) == 0: 
            #In that case just collect current
            self.collect_current_MFI(gate)
            return None
        
        #Collect MFI of markers in current gate and that of all parent gates except FSC/SSC channels
        #First initialize marker list with current gate
        if gate.xCol is None and gate.yCol is None:
            raise AliGaterError("Invalid sample encountered in collect_all_MFI, for sample "+str(self.sample)+" in gate "+str(gate.name)+" both defining markers are None\n")
        if gate.xCol is None:
            markers=[gate.yCol]
        elif gate.yCol is None:
            markers=[gate.xCol]
        else:
            markers=[gate.xCol, gate.yCol]
        
        gate_Iterator = gate
        
        while not gate_Iterator.bNoparent:
            iterator_name=gate_Iterator.name        
            iterator_parent_name = gate_Iterator.parentName
            if gate_Iterator.parent is None:
                if not gate_Iterator.bNoparent:
                    #Don't raise
                    #log warning and break
                    break
                    #raise AliGaterError("in collect_all_MFI:", "Couldn't collect MFI of "+str(iterator_name)+" due to parent population ("+str(iterator_parent_name)+") not found, is it labeled correctly in the update call?")
            gate_Iterator = self(name=iterator_parent_name)
            if not gate_Iterator: #Not found in AGSample
                #Simply not true, first update call might not be a population with None as parent!
                #Don't raise
                #log warning and break
                break
                raise AliGaterError("in collect_all_MFI:", "Couldn't collect MFI of "+str(iterator_name)+" due to parent population ("+str(iterator_parent_name)+") not found, is it labeled correctly in the update call?")
            #Save parent gates x and y markers if they are not None
            if gate_Iterator.xCol is not None:
                markers.append(gate_Iterator.xCol)
            if gate_Iterator.yCol is not None:
                markers.append(gate_Iterator.yCol)
            #Go to next parent
            #gate_Iterator = gate_Iterator.parentName
            #if gate_Iterator == 'total': #What a gate with no parent will be called
            #    break
        #Prune the markers list to unique non FSC/SSC markers
        unique_markers = list(set(markers))
        #Use collect_extra_MFI to collect MFIs for all these markers for this population
        self.collect_extra_MFI(gate=gate, MFIs_to_collect=unique_markers)
        return None
    
    def collect_extra_MFI(self, gate, MFIs_to_collect):
        #User defined list of extra MFIs to collect for current population
        markers_in_df = self.fcsDF.columns.tolist()
        for marker in MFIs_to_collect:
            if marker in markers_in_df:
                self.collect_MFI(gate, marker)
            else:
                reportStr = "Requested MFI for marker "+str(marker)+" for population "+str(gate.name)+" which doesn't exists in the sample ("+self.sample+").\n"
                sys.stderr.write(reportStr)
        return None
    
    def full_index(self):
        return list(self.fcsDF.index.values)
    
    def report(self):
        reportStr="Gating report for "+self.sample+" ("+str(len(self.fcsDF.index))+" total events)\n"+str(len(self.vGates))+" gate(s):\n"
        for gate in self.vGates:
            reportStr=reportStr+str(gate.name)+" ("+str(len(gate()))+" events)\n"
        print(reportStr)
        
    def printData(self, file, precision=4, header=True):
        if header:
            #header
            header="date\tplate\tSample\tTotal"
            for gate in self.vGates:
                header=header+"\t"+str(gate.name)+"\t"+str(gate.name)+"/"+str(gate.parentName)
            header=header+"\n"
            file.write(header)
        #Data
        reportStr=getFileName(getParent(getParent(self.filePath)))+"\t"+getFileName(getParent(self.filePath))+"\t"+self.sample+"\t"+str(len(self.full_index()))
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
        else:
            #TODO, deprecation?
            raise
                
    def resultVector(self, order=None, precision='.3f'):
        #Order is a list of marker names that must match those contained in the sample
        #It is then used to get the reported ratios in that order
        if order is not None:
            if not isinstance(order, list):
                raise TypeError("Order must be a list (of str)")
                if not all([isinstance(i,str) for i in order]):
                    raise TypeError("The elements in order must be str")
        gatesInSample = [gate[0] for gate in self.vGates]
        if order is not None:
            #Assert equal amount of specified gates as there are gates in sample
            assert len(gatesInSample) == len(order)
            #Assert the gates in the sample are the same as those in order
            assert all([population in order for population in gatesInSample])
        else:
            #If no order specified, order is gates in the sample
            order=gatesInSample
        
        orderIndex=[]
        for gateName in order:
            index = gatesInSample.index(gateName)
            orderIndex.append(index)
        assert len(orderIndex) == len(self.vGates)
        result = np.empty(shape=len(self.vGates))
        outputCounter=0
        for i in orderIndex:
            nOfEvents = len(self.vGates[i][1]())
            nOfParentEvents = len(self.vGates[i][1].getParent())

            if nOfEvents == 0 or nOfParentEvents == 0:
                fraction=float(0)
            else:
                fraction=float(format(nOfEvents/nOfParentEvents,precision))

            result[outputCounter]=fraction
            outputCounter+=1
        return result

     

class AGQC:
    """
    **Overview**
    Object that manages downsampled images while AliGater is running. It can also be used to load previously saved downsampled images to do quality control.
    
    While an AGExperiment is running with active QC the downsampled images are saved in temporary files as np arrays.
    After a run is completed they are reloaded, and stored as one bigger HDF5 object with metadata for simpler loading at a later time.
    
    **Members**
    
    TODO
    
    **Internal members**

    TODO

    **Methods**
    
    TODO

    **Internal Methods**
    
    TODO
     
    **Examples**

    None currently.
    """
    tmpFiles=[]
    downSamplingBins=32
    lGate_names=[]
    sourceFilePath=None
    h5py_filehandle=None
    image_list=[]
    sample_list=[]
    source=None
    
    def __init__(self, downSamplingBins=32, *args, **kwargs):
        self.downSamplingBins=downSamplingBins
    
    def __call__(self, fcs, *args, **kwargs):
        if len(self.tmpFiles)==0:
            sys.stderr.write("Initiating QC files\n")
        for gate in fcs.vGates:
            if gate.bQC:
                tf = self.returnTempFile(gate)
                self.printDownSample(tf, gate, fcs.sample)
    
    #Files are opened in r+w mode as per default. Call seek(0) to access for read when complete
    #Creates a tempfile for the gate if it doesn't already exists
    #Then returns that filehandle
    def returnTempFile(self, gate):
        if not str(gate.name+gate.parentName) in [tmpFile[2] for tmpFile in self.tmpFiles]:
            tmpFileDir=str(agconf.ag_tmp)
            tf = tempfile.NamedTemporaryFile(prefix="AG", dir=tmpFileDir)
            name=tf.name
            gates=str(gate.name+gate.parentName)
            self.tmpFiles.append([tf, name, gates])
            return tf
        else:
            index = [tmpFile[2] for tmpFile in self.tmpFiles].index(str(gate.name+gate.parentName))
            return self.tmpFiles[index][0]
      
    def printDownSample(self, file, gate, name):
        arr=np.array([name, str(self.downSamplingBins)])
        if gate.m_downSample is None:
            if gate.bQC:
                reportStr="WARNING: QC requested but no downsampled data for gate "+gate.name+", skipping\n"
                sys.stderr.write(reportStr)
                return
            else:
                return
        flattenedArr=gate.m_downSample.flatten()
        arr=np.append(arr,flattenedArr)
        np.save(file,arr)        
        return
    
    def reportPCs(self,folder):
        for elem in self.tmpFiles:    
            destStr=str(folder)+str(elem[2])+".AGQC.txt"
            copyfile(elem[1], destStr)
        reportStr= "Copied all downsampled files to: "+folder+"\n"
        sys.stderr.write(reportStr)
            
        for file in self.tmpFiles:
            #Create/reset list to contain images and sample names
            imlist=[]
            samplelist=[]
            #Reset file to read from beginning
            fhandle=open(file[1],"rb")
            fhandle.seek(0)
            #Read all images in the file
            while True:
                try:
                    image=np.load(fhandle)
                except OSError:
                    break
                arr=image[2:].astype(float)
                sampleName=image[0]
                downSambleBins=image[1]
                imlist.append(arr)
                samplelist.append(sampleName)
    
    def saveHDF5_QC_obj(self, destFilePath, experiment_name):
        sys.stderr.write("Creating HDF5 QC object\n")


        with h5py.File(destFilePath, 'w') as h5pyfile:
        
            for file in self.tmpFiles:
                population_name = file[2]
                population_sample_names = population_name+"_samples"
                #Create/reset list to contain images and sample names
                population_array=[]
                sample_names=[]
                #Reset file to read from beginning
                fhandle=open(file[1],"rb")
                fhandle.seek(0)
                #Read all images in the file
                while True:
                    try:
                        image=np.load(fhandle, allow_pickle=True) #CAVE: https://stackoverflow.com/questions/60191681/cannot-load-file-containing-pickled-data-python-npy-i-o
                    except OSError:
                        break
                    #Image data is from third element and onwards
                    arr=image[2:].astype(float)
                    population_array.append(arr)
                    #Resolution (int) is at position 2
                    downSamplingBins=image[1].astype(int)
                    #Sample name is at position 0, we recode to UTF-8 for H5py compatibility
                    sample_name=image[0].encode('UTF-8')
                    sample_names.append(sample_name)

                dataset = h5pyfile.create_dataset(population_name, data=population_array)
                dataset.attrs['resolution']=downSamplingBins
                dt = h5py.special_dtype(vlen=str)
                dataset = h5pyfile.create_dataset(population_sample_names, data=sample_names, dtype=dt)
                
        reportStr="Saved QC data to: "+str(destFilePath)+"\n"
        sys.stderr.write(reportStr)
    
    def load_QC_file(self, sourceFilePath):
        self.sourceFilePath = sourceFilePath
        with h5py.File(sourceFilePath, 'r') as h5pyfile:
            all_keys = list(h5pyfile.keys())
            for key in all_keys:
                if not key[-8:] == '_samples':
                    self.lGate_names.append(key)
            reportStr="Loaded metadata for QC file with "+str(len(self.lGate_names))+" gates:\n"
            sys.stderr.write(reportStr)
            for gate_name in self.lGate_names:
                sys.stderr.write(gate_name)
                sys.stderr.write("\n")
            sys.stderr.write("Load a specific population for QC with select_population\n")
            self.h5py_filehandle = h5pyfile
    
    def image_viewer(self, sample_data=None, image_data=None, name=None, rand_select=None, mask_zero=False):
        if sample_data is None:
            sample_data = self.sample_list
        if image_data is None:
            image_data = self.image_list
        assert len(sample_data) == len(image_data)
        #Check if a name is given first
        if name is not None:
            if not isinstance(name,str):
                raise AliGaterError("in AGQC.image_viewer", "name must be str")
            for i in np.arange(0,len(sample_data),1):
                if name.lower() == sample_data[i].lower():
                    index=i
                    break
            else:
                reportStr="sample "+str(name)+" not found in passed arrays."
                sys.stderr.write(reportStr)
                return None
            plot_flattened_heatmap(image_data[index], self.downSamplingBins, mask=mask_zero)
        elif rand_select is not None:
            if not isinstance(rand_select,int):
                raise AliGaterError("in AGQC.image_viewer", "rand_select must be int")
            if rand_select >= len(image_data):
                sys.stderr.write("WARNING: random selection larger than there is available images")
                rand_select = len(image_data)
            random_selection = np.random.randint(len(image_data), size=rand_select)
            for i in random_selection:
                print(sample_data[i])
                plot_flattened_heatmap(image_data[i], self.downSamplingBins, mask=mask_zero)
            return None
        else:
            for index in np.arange(0,len(image_data),1):
                print(sample_data[index])
                plot_flattened_heatmap(image_data[index], self.downSamplingBins, mask=mask_zero)   
        return None
    
    def select_population(self, population, return_type="list"):
        if not return_type.lower() in ['nparray','list']:
            raise AliGaterError("in select_population"," return type must be either 'list' or 'nparray'")
        if self.sourceFilePath is None:
            reportStr="No QC object is loaded!\nRun load_QC_file first\n"
            sys.stderr.write(reportStr)
            return None
        with h5py.File(self.sourceFilePath, 'r') as h5pyfile:
            all_keys = list(h5pyfile.keys())
            for key in all_keys:
                if key.lower() == population.lower():
                    match = key
                    match_samples = match+"_samples"
                    break
            else:
                reportStr="population not found in QC obj: "+str(population)+"\nYou can view available gates in currently loaded QC obj with <QCOBJ>.lGate_names\n"
                sys.stderr.write(reportStr)
                return None
            image_data = h5pyfile[match][:]
            sample_data = h5pyfile[match_samples][:]
            self.downSamplingBins = h5pyfile[match].attrs['resolution']
            if not len(image_data)==len(sample_data):
                raise AliGaterError("invalid AGQC file"," not equal number of sample names and sample images")
            reportStr="Loaded data for "+str(population)+" containing "+str(len(sample_data))+" ("+str(self.downSamplingBins)+","+str(self.downSamplingBins)+") images\n"
            sys.stderr.write(reportStr)
            self.sample_list = sample_data
            self.image_list = [x.reshape(self.downSamplingBins, self.downSamplingBins) for x in image_data]
            if return_type.lower() == 'nparray':
                sample_data = np.asarray(sample_data)
                image_data = np.asarray(image_data)
            else:
                sample_data = self.sample_list
                image_data = self.image_list
            return sample_data, image_data

    def select_images(self, imlist):
        if self.sample_list is None:
            reportStr="No population is selected!\nRun select_population first\n"
            sys.stderr.write(reportStr)
            return None
        if not isinstance(imlist, (list, np.ndarray)):
            raise AliGaterError("in AGQC::select_images","passed imlist must be list of str sample names")
            if not all(isinstance(x, str) for x in imlist):
                raise AliGaterError("in AGQC::select_images","passed imlist must be list of str sample names")
        output_images=[]
        output_samples=[]
        not_found=[]
        nFound=0
        for i in np.arange(0,len(imlist),1):
            for ix in np.arange(0,len(self.sample_list),1):
                if imlist[i].lower() == self.sample_list[ix].lower():
                    output_images.append(self.image_list[ix])
                    output_samples.append(self.sample_list[ix])
                    nFound+=1
                    break
            else:
                not_found.append(imlist[i])
        if len(not_found) > 0 and len(not_found) != len(imlist):
            reportStr=str(len(not_found))+" image(s) not found in currently loaded population:\n"
            sys.stderr.write(reportStr)
            for missing_samp in not_found:
                sys.stderr.write(missing_samp)
                sys.stderr.write("\n")
        if len(not_found) == len(imlist):
            sys.stderr.write("No images found\n")
        return output_samples, output_images
        
    def define_training_sets(self, lSamplelists, shuffle=True):
        if self.sample_list is None:
            reportStr="No population is selected!\nRun select_population first\n"
            sys.stderr.write(reportStr)
            return None
        if not isinstance(lSamplelists,(list, np.ndarray)):
            raise AliGaterError("in AGQC::define_training_sets","passed lSamplelists must be list/numpy array")
        if not all(isinstance(x,(list,np.ndarray)) for x in lSamplelists):
            raise AliGaterError("in AGQC::define_training_sets","lSamplelists must contain >2 lists/numpy arrays with sample names")
        
        curClass = 0
        full_train_array=[]
        for training_set in lSamplelists:
            current_set_samples = self.select_images(training_set)
            current_train_array=[]
            for i in np.arange(0, len(current_set_samples[0]), 1):
                current_train_array.append([curClass,current_set_samples[0][i],current_set_samples[1][i]])
            full_train_array = full_train_array + current_train_array
            curClass+=1
        
        if shuffle:
            np.random.shuffle(full_train_array)
        target_array = [x[0] for x in full_train_array]
        name_array = [x[1] for x in full_train_array]
        im_array = [x[2] for x in full_train_array]
        
        return name_array, target_array, im_array
        
    def loadDownSampleNumpyArr(self,file):
        #Create/reset list to contain images and sample names
        imlist=[]
        samplelist=[]
        #Reset file to read from beginning
        fhandle=open(file[1],"rb")
        fhandle.seek(0)
        #Read all images in the file
        while True:
            try:
                image=np.load(fhandle)
            except OSError:
                break
            arr=image[2:].astype(float)
            sampleName=image[0]
            imlist.append(arr)
            samplelist.append(sampleName)
        return [samplelist, imlist]
    
    
    def run_QC(self, population, filter_samples=[], n_components=2):
        if self.sourceFilePath is None:
            reportStr="No QC object is loaded!\nRun load_QC_file first\n"
            sys.stderr.write(reportStr)
            return None
        samplist,imlist = self.select_population(population, return_type="nparray")
        if not isinstance(n_components,int):
            raise AliGaterError("in AGQC::run_QC", "n_components must be int, found"+str(type(n_components)))
        if not isinstance(filter_samples,list):
            raise AliGaterError("in AGQC::run_QC", "filter_samples must be list of str, found"+str(type(n_components)))
        if not all([isinstance(x,str) for x in filter_samples]):
            raise AliGaterError("in AGQC::run_QC", "filter_samples must be list of str, the list contained non-str entries")
        if len(filter_samples)>=1:
            include_indicies=[]
            removed_samples=[]
            for i in np.arange(0,len(samplist),1):
                #if samplist[i] in filter_samples:
                str_samp=str(samplist[i]) #str comparison for below to work, CAVE: might be slow
                if any([str(samp_to_filter) in str_samp for samp_to_filter in filter_samples]):
                    removed_samples.append(samplist[i])
                else:
                    include_indicies.append(i)
            filtered_samplist = samplist[include_indicies]
            filtered_imlist = imlist[include_indicies]
            PC_DF = imagePCA_cluster(samplelist=filtered_samplist,imlist=filtered_imlist, nOfComponents=n_components)
            reportStr="After filtering, "+str(len(removed_samples))+" were removed (filter size: "+str(len(filter_samples))+").\n"
            sys.stderr.write(reportStr)
        else:
            PC_DF = imagePCA_cluster(samplelist=samplist,imlist=imlist, nOfComponents=n_components)
        return PC_DF
    

    
class AGExperiment:
    """
    **Overview**

    The AGExperiment object is where you define several rules and options for your analysis. 
    It orchestrates how your custom strategy will interact with the flow data and handles optional settings like automatic QC. 
    
    **Members**
    
    filters, list-like
        List-like of str. Inclusive filter that will be passed on to AGFileSystem.CollectFiles.
        
    ignoretype, list-like
        List-like of str. Exclusive filter that will be passed on to AGFileSystem.CollectFiles.
        
    mask, list-like
        List-like of str. Exclusive filter that will be passed on to AGFileSystem.CollectFiles.
        
    markers, list-like
        List-like of str. Both inclusive and exclusive filter that will be passed on to AGFileSystem.loadFCS. 
        Any marker labels encountered in a .fcs file must exist in this filter, otherwise sample is skipped.
        
        NOTE: Forwardscatters, sidescatters and time should not be included.
    
    compensation_exceptions, list-like
        List like of str. Contains absolute filepaths to .fcs files that should be compensated in a different way than the rest of the samples.
    
    comp_matrix, ndarray
        Optional comp_matrix that can be passed and used for applying compensation manually.

    resultHeader, list-like
        List-like of str. A custom header to be used for the output file.
        
    bQC, bool 
        Flag if any type of QC has been requested. This will automatically trigger post-analysis PCA, print figures and save a pandas DataFrame with the results.

    nOfEvents, int, default : None
        Number of maximum events to be collected from each fcs file in experiment. Default is None, which means all events are read from the .fcs file.

    **Internal members**

    fcsList, list-like
        List like of str filepath. Collected through an internal call to AGFileSystem.CollectFiles
        
    sampleList, list-like
        List-like of AGClasses.AGSample objects. Created and expanded as a batch experiment runs.
        Note that these are not fully-fletched AGSampled object, but rather slimmed versions, with 
        only enough information to recreate an identical sample object to save memory.
    
    man_comp, bool
        Flag recognizing if a sample should be compensated manually with the compensation matrix stored in comp_matrix.

    flaggedSamples, list-like
        List-like of list-like. Two-by-n list with [samplename, flagged_population].
    
    resultMatrix, list-like
        Matrix with reported statistics, one sample per row.

    normaliseLevel, str
        Experimental feature, for plate normalisation. Not fully implemented.
    
    nOfPlates, int
        Count of plates, used for plate normalisation. Not fully implemented.
    
    normalise, bool
        Flag for plate normalisation. Not fully implemented.
    
    plateList, list-like
        List-like of str filepaths. Used for plate normalisation. Not fully implemented.
    
    HDF5Fileset, bool
        Flag for loading pre-processed HDF5 datasets created with create_HDF5_files.
    
    **Methods**
    
    apply(strategy, \*args, \*\*kwargs)
        Main method where you apply your custom strategy to all .fcs files in an experiment.
        
        Any arguments defined in \*args or \*\*kwargs will be passed on to your strategy, this way you can pass on and parse custom arguments to your strategy function.
        See sample notebooks for how to build a strategy.
        
        NOTE: The AGExperiment object checks if the passed strategy seems to be a function by checking for a __call__ attribute, this means that you can pass a function, a class or similar as a strategy.
        The requirement other than that is that the strategy should accept an AGSample object and return an AGSample object.
        
    **Internal Methods**
    
    TODO
    
    **Examples**

    None currently.
    
    **Notes**
    
    On AGSample objects:
        
    | The sample object for each .fcs file only exists while it's being gated. 
    | After a strategy has been applied the populations are stored in a less memory intensive way and the sample object deconstructed. 
    | This means that you cannot access the same AGSample object that was gated previously. However, if set up properly, the AGExperiment can recreate an identical AGSample object quickly.
 
    
    """
    fcsList=[]              #LIST (str filepaths)
    normaliseLevel=None     #STR
    nOfPlates=0             #INT    
    normalise=None          #BOOL
    plateList=[]            #LIST (str filepaths)
    sampleList=[]           #LIST OF AGSample OBJECTS
    lFilter=None
    lMask=None
    lIgnoreTypes=None
    lMarkers=None
    comp_matrix=None
    man_comp=False
    compensation_exceptions=None
    flaggedSamples=[]       #List of tuple (path, flagged pop)
    resultHeader=None
    resultMatrix=None
    result_MFI_DF = None
    has_MFI = False
    output_folder=None
    exp_name=None
    bQC=False
    flourochrome_area_filter=False
    QCbins=32
    HDF5Fileset=False
    nOfEvents=None
    
    def __init__(self, experimentRoot, *args, **kwargs):
        if 'filters' in kwargs:
            if not isinstance(kwargs['filters'],str):
                if not isinstance(kwargs['filters'],list):
                    raise TypeError("if filters is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['filters']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in filters")
                    self.lFilter=kwargs['filters']
            else:
                self.lFilter=[kwargs['filters']]
            reportStr=str(len(self.lFilter))+" filter(s) defined\n"
            sys.stderr.write(reportStr)
            
        if 'mask' in kwargs:
            if not isinstance(kwargs['mask'],str):
                if not isinstance(kwargs['mask'],list):
                    raise TypeError("if mask is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['mask']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in mask")
                    self.lMask=kwargs['mask']
            else:
                self.lMask=[kwargs['mask']]
            reportStr=str(len(self.lMask))+" mask(s) defined\n"
            sys.stderr.write(reportStr)
            
        if 'ignoretype' in kwargs:
            if not isinstance(kwargs['ignoretype'],str):
                if not isinstance(kwargs['ignoretype'],list):
                    raise TypeError("if ignoretype is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['ignoretype']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in mask")
                    self.lIgnoreTypes=kwargs['ignoretype']
            else:
                self.lIgnoreTypes=[kwargs['ignoretype']]
            reportStr=str(len(self.lIgnoreTypes))+" ignoretype defined\n"
            sys.stderr.write(reportStr)
            
        if 'markers' in kwargs:
            if not isinstance(kwargs['markers'],str):
                if not isinstance(kwargs['markers'],list):
                    raise TypeError("if markers is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['markers']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in markers")
                    self.lMarkers=kwargs['markers']
            else:
                self.lMarkers=[kwargs['markers']]

        if 'flourochrome_area_filter' in kwargs:
            if not isinstance(kwargs['flourochrome_area_filter'],bool):
                raise TypeError("flourochrome_area_filter must be a bool (True/False).")
            else:
                self.flourochrome_area_filter=[kwargs['flourochrome_area_filter']]
        
        if 'nOfEvents' in kwargs:
            if not isinstance(kwargs['nOfEvents'],int):
                raise TypeError("nOfEvents has to be an integer.")
            elif kwargs['nOfEvents'] <= 0:
                raise TypeError("nOfEvents has to be a positive integer")
            else:
                if kwargs['nOfEvents'] < agconf.cellFilter:
                    raise AliGaterError("In AGExperiment","Experiment object initialized with nOfEvents parameter less than the cellFilter specified in AGConfig")
                self.nOfEvents=kwargs['nOfEvents']
        
        if 'normaliseOn' in kwargs:
            if not isinstance(kwargs['normaliseOn'],str):
                raise (TypeError("normaliseOn must be string"))
            self.normalise=True
            self.normaliseLevel=kwargs['normaliseOn']
            reportStr="Plate normalisation/outlier detection requested\nNOT IMPLEMENTED YET\n"
            sys.stderr.write(reportStr)
            
        if 'QC' in kwargs:
            if not isinstance(kwargs['QC'],bool):
                raise (TypeError("QC must be bool (True/False)"))
            self.bQC=kwargs['QC']
            if self.bQC:
                reportStr="QC requested\n"
                sys.stderr.write(reportStr)
        
        if 'QCbins' in kwargs:
            if self.bQC:
                if not isinstance(kwargs['QCbins'],int):
                    raise (TypeError("QCbins must be int"))
                self.QCbins=kwargs['QCbins']
            else:
                raise(ValueError("QCbins specified but QC not requested"))
        
        if 'compMatrix' in kwargs:
            if not isinstance(kwargs['compMatrix'],np.ndarray):
                raise TypeError("compensation matrix must be np array if passed")
            self.comp_matrix=kwargs['compMatrix']
            self.man_comp=True
            reportStr="Manual compensation specified\n"
            sys.stderr.write(reportStr)
        
        if 'compList' in kwargs:
            if not isinstance(kwargs['compList'],list):
                raise TypeError("compList must be passed as list of tuples (file, file with compensation)")
            else:
                for elem in kwargs['compList']:
                    if not isinstance(elem, tuple):
                        raise TypeError("elements in complist must be tuples (file, file with compensation)")
            if self.man_comp:
                raise ValueError("Cannot both give compensation exception list and provide manual compensation matrix")
            self.compensation_exceptions=kwargs['compList']
        
        if 'experiment_name' in kwargs:
            if not isinstance(kwargs['experiment_name'],str):
                raise (TypeError("experiment_name must be specified as string"))
            try:
                self.exp_name = kwargs['experiment_name']
                self.output_folder=str(agconf.ag_out)+"/"+self.exp_name
                if check_filePath(self.output_folder) != "dir":
                    os.makedirs(self.output_folder)
            except FileExistsError:
                reportStr="WARNING: specified output directory ("+str(agconf.ag_out)+str(kwargs['experiment_name'])+") already exists, content in folder might be overwritten without warning\n"
                sys.stderr.write(reportStr)
                pass
            
        if 'HDF5Fileset' in kwargs:
            if not isinstance(kwargs['HDF5Fileset'],bool):
                raise TypeError("HDF5Fileset must be specified as True/False")
            self.HDF5Fileset = kwargs['HDF5Fileset']
        
        if self.output_folder is None:
            self.exp_name = 'AGexperiment_'+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '_')
            self.output_folder=str(agconf.ag_out)+"/"+self.exp_name
            reportStr="No experiment name specified, generated name: "+self.exp_name+"\n"
            sys.stderr.write(reportStr)
            if check_filePath(self.output_folder) != "dir":
                os.makedirs(self.output_folder)
        
        #Check that output folder path variable ends with a slash, no matter if its generated or manually specified
        #And save it as a class member
        if self.output_folder[-1] != "/":
            self.output_folder = self.output_folder+"/"
        else:
            self.output_folder = self.output_folder
        
        
        if not isinstance(experimentRoot,str):
            if not isinstance(experimentRoot, list):
                raise AliGaterError("in AGExperiment initialisation: ","ExperimentRoot must be str filepath or list of str filepaths")
            else:
                bList=True
        else:
            bList=False
            
        #If no normalisation, collect all files
        if not bList:
            self.fcsList=collectFiles(experimentRoot,self.lFilter,self.lMask,self.lIgnoreTypes, self.HDF5Fileset)
        else:
            sys.stderr.write("Experiment initialised with file list. Checking entries...\n")
            file_list=check_exists(experimentRoot)
            lFlaggedIndicies=applyFilter(file_list, self.lFilter,self.lMask, self.lIgnoreTypes)
            lOutput = [i for j, i in enumerate(file_list) if j not in lFlaggedIndicies]
            sOutputString="Collected "+str(len(lOutput))+" files, "+str(len(lFlaggedIndicies))+" files did not pass filter(s) and mask(s).\n"
            sys.stderr.write(sOutputString)
            self.fcsList=lOutput
            
        if self.normalise:
            #Otherwise (plate normalisation) collect abs path to all end-of-tree folders
            self.plateList=listDir(experimentRoot)
            self.checkPlateList()
            reportStr="Files are distributed in "+str(len(self.plateList))+" plates\n"
            sys.stderr.write(reportStr)
        
        
    def checkPlateList(self):
        #Function that goes through the list of folders and removes those that do not contain fcs files
        #The remaining folders are considered plates
        lFlaggedFolders=[]
        for index,folder in enumerate(self.plateList):
            filesInFolder=next(os.walk(folder))[2]
            if not filesInFolder:
                lFlaggedFolders.append(index)
        #To avoid cascading indicies while deleting go from last element->first
        lFlaggedFolders=np.sort(lFlaggedFolders)[::-1]
        for index in lFlaggedFolders:
            del self.plateList[index]
    
    def apply(self, strategy, n_ray_workers=1, *args, **kwargs):
        if not hasattr(strategy, '__call__'):
            raise TypeError("Passed strategy does not seem to be a function")
        if self.bQC:
            QCObj=AGQC(downSamplingBins=self.QCbins)
        
        for fcs in self.fcsList:
            if 'ag_verbose' in kwargs:
                agconf.ag_verbose = kwargs['ag_verbose']
            if self.HDF5Fileset:
                sample = loadHDF5sample(fcs, sampling_resolution=self.QCbins)
            else:
                if self.man_comp:
                    comp_matrix=self.comp_matrix
                else:
                    comp_matrix=None
                if self.compensation_exceptions is not None:
                    comp_matrix=self.check_compensation_exception(fcs)

                sample = loadFCS(fcs, return_type="AGSample", comp_matrix=comp_matrix, markers=self.lMarkers, flourochrome_area_filter=self.flourochrome_area_filter, sampling_resolution=self.QCbins, nOfEvents = self.nOfEvents)
            if sample is None:
                continue
            sys.stderr.write("Applying strategy\n")
            gatedSample=self.fcs_apply_strategy(sample,strategy, *args, **kwargs)
            if self.resultHeader is None:
                    self.initResultMatrix(sample)  #BIG WARNING IN DOC FOR THIS BEHAVIOR
            self.collectGateData(gatedSample)
            self.sampleList.append(gatedSample.sample)
            if self.bQC:
                QCObj(gatedSample)
            sys.stderr.write("Sample gating ")
            if self.bQC:
                sys.stderr.write("and QC metrics collection ")
            sys.stderr.write("done\n")
            sys.stderr.flush()
                    
        nOfFlagged=len(self.flaggedSamples)
        if "/" in self.exp_name:
            fixed_exp_name=self.exp_name.split('/')[-1]
        else:
            fixed_exp_name=self.exp_name
        log_file_name = self.output_folder+fixed_exp_name+".log.txt"
        fhandle=open(log_file_name,'w')
        reportStr="Complete, "
        if nOfFlagged>0:
            reportStr=reportStr+str(nOfFlagged)+" samples had at least one gate with an invalid flag set"
        else:
            reportStr=reportStr+"no samples had populations with invalid flags"
        reportStr=reportStr+"\n"
        sys.stderr.write(reportStr)
        fhandle.write(reportStr)
        for elem in self.flaggedSamples:
            reportStr=elem+"\n"
            sys.stderr.write(reportStr)
            fhandle.write(reportStr)
        if self.bQC:
            #if 'folder' in kwargs:
            #    folder=kwargs['folder']
            #TODO: default folder
            #QCObj.reportPCs(self.output_folder)
            if "/" in self.exp_name:
                fixed_exp_name=self.exp_name.split('/')[-1]
            else:
                fixed_exp_name=self.exp_name
            QC_file_name=self.output_folder+fixed_exp_name+".QC.HDF5"
            QCObj.saveHDF5_QC_obj(destFilePath=QC_file_name, experiment_name=self.exp_name)

    
    def initResultMatrix(self, fcs):
        assert len(fcs.vGates) > 0
        self.resultMatrix=[]
        self.resultHeader=["sampleName"]
        for gate in fcs.vGates:
            if gate.name==None or gate.parentName is None:
                raise AliGaterError("Error when initializing resultmatrix gatename or parentname missing")
            self.resultHeader.extend([gate.name, gate.name+str("/")+gate.parentName])
        
        if len(fcs.MFI_Series) > 0 :
            #if MFI available, also init MFI table
            self.result_MFI_DF = pd.DataFrame()
            self.has_MFI = True
            
    def collectGateData(self, fcs):
        assert all(isinstance(i, list) for i in [self.resultHeader, self.resultMatrix])
        #Create empty python list of correct size
        sampleResults=[None]*len(self.resultHeader)
        sampleResults[0] = fcs.sample
        bFlagged=False
        for gate in fcs.vGates:
            try:
                indexOfGate=self.resultHeader.index(gate.name)
            except ValueError:
                reportStr="sample contain unknown gate: "+gate.name+", skipping gate\n"
                sys.stderr.write(reportStr)
                sys.stderr.write(fcs.filePath)
                continue
            currentGate=float(len(gate.current))
            if gate.parent is not None:
                currentGateParent=len(gate.parent)
            else: 
                currentGateParent=len(fcs.full_index())
            if currentGateParent==0:
                currentRatio="NA"
            else:
                currentRatio=currentGate/currentGateParent
            if gate.bInvalid:
                bFlagged=True
                sampleResults[indexOfGate]=currentGate
                sampleResults[indexOfGate+1]=currentRatio
            elif gate.bRatioGate:
                sampleResults[indexOfGate]="NA"
                sampleResults[indexOfGate+1]=currentRatio
            else:
                sampleResults[indexOfGate] = currentGate
                sampleResults[indexOfGate+1] = currentRatio
        self.resultMatrix.append(sampleResults)
        
        #****Check/Add MFI*****
        if len(fcs.MFI_Series) > 0:
            #Make sure duplicate entries are dropped (can happen if extra MFIs are applied etc)
            non_dup_series=fcs.MFI_Series.drop_duplicates()
            #Make sure the series has the correct name
            non_dup_series.name=fcs.sample
            
            dups = non_dup_series[non_dup_series.index.duplicated()]
            if len(dups) > 0:
                #This will cause a worse pandas error if allowed ('cannot reindex from duplicate axis') 
                sys.stderr.write("Duplicated entry in MFI data, have you incorrectly labeled some position? Printing offending labels:\n")
                for index,elem in dups.iteritems():
                    reportStr = str(index)+"\t"+str(elem)+"\n"
                    sys.stderr.write(reportStr)
                raise AliGaterError("Duplicated entry in MFI data.")
            # for index,entry in non_dup_series.iteritems():
            #     reportStr=str(index)+"\t"+str(entry)+"\n"
            #     sys.stderr.write(reportStr)
            #Append
            self.result_MFI_DF = self.result_MFI_DF.append(non_dup_series)
        
        if bFlagged:
            self.flaggedSamples.append(fcs.sample)
    
    def printExperiment(self, file=None, MFI_file=None):
        if file is None:
            file=self.fallback_filepath()
        else:
            if not isinstance(file,str) or not (check_filePath(file) in ["dir_exists", "file"]):
                #filepath should be string path to output file. Must either exist, or it's directory must exist
                sys.stderr.write("Warning, in aligater.AliGaterExperiment.printExperiment: file filepath invalid, falling back to ag_out directory.")
                file=self.fallback_filepath()
        
        
        if not all(isinstance(i, list) for i in [self.resultHeader, self.resultMatrix]):
            sys.stderr.write("Experiment data table empty, no results to print.\n")
            #No ratio/count data, but maybe MFI?
            if not self.has_MFI:
                return None
        else:
            fhandle = open(file, 'w')
            for elem in self.resultHeader:
                outputStr=str(elem)+"\t"
                fhandle.write(outputStr)
            fhandle.write("\n")
            for sample in self.resultMatrix:
                for elem in sample:
                    outputStr=str(elem)+"\t"
                    fhandle.write(outputStr)
                fhandle.write("\n")
            fhandle.close()
        
        #MFI stuff
        if self.has_MFI:
            sys.stderr.write("printing MFI data\n")
            if len(self.result_MFI_DF) == 0:
                sys.stderr.write("No MFI data to print.\n")
                return None
                MFI_file=file=self.fallback_filepath(".MFI")
            else:
                if not isinstance(MFI_file,str) or not (check_filePath(MFI_file) in ["dir_exists", "file"]):
                    #filepath should be string path to output file. Must either exist, or it's directory must exist
                    sys.stderr.write("Warning, in aligater.AliGaterExperiment.printExperiment: MFI_file filepath invalid, falling back to ag_out directory.")
                    MFI_file = self.fallback_filepath(".MFI")

            self.result_MFI_DF.to_csv(MFI_file, sep="\t")
          
    def fallback_filepath(self, suffix=""):
        fixed_exp_name=self.exp_name.split('/')[-1]
        fallback_path = self.output_folder+fixed_exp_name+suffix+".results.txt"
        return fallback_path
    
    def parse_folder(self, strategy, folder, comp_matrix=None, *args, **kwargs):
        #NOT USED/WORK IN PROGRESS
        if 'ag_verbose' in kwargs:
            agconf.ag_verbose = kwargs['ag_verbose']

        reportStr="Parsing plate: "+folder+" \n"
        sys.stderr.write(reportStr)
        fcs_in_folder = collectFiles(folder, lFilter=self.lFilter, lMask=self.lMask, lIgnoreTypes=self.lIgnoreTypes, HDF5 = self.HDF5Fileset)
        samples_in_plate=[]
        for fcs in fcs_in_folder:
            if self.man_comp:
                comp_matrix=self.comp_matrix
            if self.compensation_exceptions is not None:
                    comp_matrix=self.check_compensation_exception(fcs)
            sample = loadFCS(fcs, return_type="AGSample", comp_matrix=comp_matrix, markers=self.lMarkers)
            if sample is None:
                continue
            sample = self.fcs_apply_strategy(sample, strategy, *args, **kwargs)
            if self.resultHeader is None:
                self.initResultMatrix(sample)
            sample.fcsDF=[]
            samples_in_plate.append(sample)
        return samples_in_plate
    
    def fcs_apply_strategy(self, fcs, strategy, *args, **kwargs):
        #Check for strategy __Call__ functionality
        fcs = strategy(fcs, *args, **kwargs)
        return fcs
    
    
    def check_metadata_internal(self, filePath, fcsDF, lFlagged, metaDict):
        bOk=True
        lFlags=[]
        if 'SPILL' in metaDict.keys():
            spill_keyword='SPILL'
        elif '$SPILL' in metaDict.keys():
            spill_keyword='$SPILL'
        elif 'SPILLOVER' in metaDict.keys():
            spill_keyword='SPILLOVER'
        elif '$SPILLOVER' in metaDict.keys():
            spill_keyword='$SPILLOVER'
        else:
            reportStr="Unknown spillover keyword in metadata"
            lFlags.append(reportStr)
            bOk=False

        spill_matrix=metaDict[spill_keyword].split(',')
        
        if len(fcsDF) < agconf.cellFilter:
            reportStr="Sample has fewer events than cellFilter treshold"
            lFlags.append(reportStr)
            bOk=False
        
        n = int(spill_matrix[0]) #number of colors
        columns=fcsDF.columns.tolist()
        exclude_cols = ['tlsw',	'tmsw', 'event info','time']
        indicies_to_drop=[]
        fsc_ssc_count=0
        for index,col in enumerate(columns):
            bExclusion=False
            for exclusion in exclude_cols:
                if exclusion in col.lower():
                    indicies_to_drop.append(index)
                    bExclusion=True
            if 'fsc' not in col.lower() and 'ssc' not in col.lower():
                if col.lower()[-2:] != '-a' and not bExclusion and self.flourochrome_area_filter:
                    indicies_to_drop.append(index)
                    continue
            else:
                fsc_ssc_count += 1
        fcsDF.drop(fcsDF.columns[indicies_to_drop], axis=1, inplace=True)
        colNames = fcsDF.columns        
        #Check that the fcs has a supported fcs version
        try:
            fcs_vs = metaDict['__header__']['FCS format'].decode('UTF-8')
        except UnicodeError:
            reportStr="Unexpected encoding in metadata (not UTF-8), proceed with caution"
            bOk=False
            lFlags.append(reportStr)
        if not fcs_vs in ['FCS3.0','FCS2.0','FCS3.1']:
            reportStr="FCS file version ("+fcs_vs+") not guaranteed to work, procced with caution"
            bOk=False
            lFlags.append(reportStr)
            
        #Check that the first four fields in the fcs are forward and side scatters
        #Deprecated, uneccessary check
        # if not self.flourochrome_area_filter:
        #     scatter_cols=fcsDF.columns[0:4]
        #     if not all([scatter in ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H'] for scatter in scatter_cols]):
        #         reportStr="First four columns in fcs are not forward and side scatters"
        #         bOk=False
        #         lFlags.append(reportStr)
        # else:
        #     scatter_cols=fcsDF.columns[0:5]
        #     if not all([['FSC', 'SSC'] in scatter for scatter in scatter_cols]):
        #         reportStr="First four columns in fcs are not forward and side scatters"
        #         bOk=False
        #         lFlags.append(reportStr)
            
        #If markers have been passed to the experiment object, check that they are
        #correctly specified for each sample
        if self.lMarkers is not None:
            if not set(colNames) == set(self.lMarkers):
                reportStr="Specified markers not in fcs"
                bOk=False
                lFlags.append(reportStr)
                
        #Check that compensation data is available (non-zero)
        comp_matrix = np.array(spill_matrix[n+1:]).reshape(n, n).astype(float)
        tmp_identity = np.identity(n)
        if np.array_equal(comp_matrix, tmp_identity):
            reportStr="No compensation data available"
            bOk=False
            lFlags.append(reportStr)
            
        if not bOk:
            lFlags.append(filePath)
            lFlagged.append(lFlags[::-1])  
        return bOk
    
    def check_metadata(self, *args, **kwargs):
        if len(self.fcsList) == 0:
            raise ValueError("This AGExperiment does not have any fcs files associated with it")
        bOk=True
        metadata_warnings=0
        reportStr="Checking metadata for all "+str(len(self.fcsList))+" fcs files in experiment:\n"
        sys.stderr.write(reportStr)
        lFlagged=[]
        progress_counter=0
        for fcs in self.fcsList:
            progress_counter+=1
            metadata,fcsDF = parse(fcs,output_format='DataFrame')
            if metadata is None:
                metadata_warnings+=1
                lFlagged.append([fcs,"Could not parse metadata"])
                continue
            bOk=self.check_metadata_internal(fcs,fcsDF,lFlagged,metadata)
            if not bOk:
                metadata_warnings+=1
            if progress_counter%10==0:
                progress=progress_counter/len(self.fcsList)
                update_progress(progress)
        update_progress(1.0)
        reportStr=str(metadata_warnings)+" files had metadata flags\n"
        sys.stderr.write(reportStr)   
        if len(lFlagged) > 0:
            for sample in lFlagged:
                reportStr=sample[0]+"\nFlags:\n"
                sys.stderr.write(reportStr)
                for i in np.arange(1,len(sample),1):
                    reportStr=sample[i]+"\n"
                    sys.stderr.write(reportStr)
        return lFlagged            
    
    def check_compensation_exception(self, fcs):
        for i in np.arange(0, len(self.compensation_exceptions),1):
            if fcs.lower() == self.compensation_exceptions[i][0].lower():
                reportStr="Sample ("+str(fcs)+") is in compensation exception list, collecting external compensation matrix\n"
                sys.stderr.write(reportStr)
                try:
                    metadata, fcsDF = loadFCS(self.compensation_exceptions[i][1], return_type="index", metadata=True,ignore_minCell_filter=True)
                except FileNotFoundError:
                    reportStr="WARNING, file for collecting external compensation matrix does not exist. Could not retrieve the external compensation matrix for this sample\n"
                    sys.stderr.write(reportStr)
                    return None
                comp_matrix=getCompensationMatrix(fcsDF, metadata)[1]
                if comp_matrix is not None:
                    reportStr="Succesfully collected external compensation matrix\n"
                    sys.stderr.write(reportStr)
                    return comp_matrix
                else:
                    reportStr="WARNING, in check_compensation_exception: external compensation sample "+self.compensation_exceptions[i][1]+" had a zero or none compensation matrix\n"
                    sys.stderr.write(reportStr)
                    self.flaggedSamples.append((fcs,"NO EXTERNAL COMP"))
        return None
    
    def create_HDF5_files(self, target_root_dir, folder_naming_levels=0):
        reportStr="Creating compressed compensated files for all fcs files loaded in the experiment.\nMetadata and filter checks will be limited, it is strongly recommended to run check_metadata first and update experiment file filters and masks in the AGExperiment accordingly.\nRoot folder for binary files: "+str(target_root_dir)+"\n"
        sys.stderr.write(reportStr)
        bFolderExists = os.path.isdir(target_root_dir)
        if not bFolderExists:
            sys.stderr.write("target_root_dir doesn't exists, attempting to create\n")
            os.mkdir(target_root_dir)

        #Load FCSes here
        for fcs in self.fcsList:
            if self.man_comp:
                comp_matrix=self.comp_matrix
            else:
                comp_matrix=None
            if self.compensation_exceptions is not None:
                comp_matrix=self.check_compensation_exception(fcs)
            sample = loadFCS(fcs, comp_matrix=comp_matrix, compensate=True, markers=self.lMarkers, flourochrome_area_filter=self.flourochrome_area_filter, return_type="agsample")
            if sample is None:
                continue

            tmp_filePath=sample.filePath
            if folder_naming_levels > 0:
                parent_folder_names=[]
                parent_folder_paths=[]
                for i in np.arange(0, folder_naming_levels,1):
                    parentFolder = getParent(tmp_filePath) 
                    parent_folder_paths.append(parentFolder)
                    parent_folder_names.append(getFileName(parentFolder))
                    tmp_filePath = parentFolder

            target_folder=target_root_dir
            for folder in parent_folder_names[::-1]:
                if target_folder[-1] != "/":
                    target_folder=target_folder+"/"+folder
                else:
                    target_folder=target_folder+folder
                if not os.path.isdir(target_folder):
                    os.mkdir(target_folder)
                    
            #Subfolders exists construct h5py binary filename
            h5py_filehandle=target_root_dir+"/".join(parent_folder_names[::-1])+"/"+getFileName(sample.filePath)+".h5"
            h5py_internal_name = "/".join(parent_folder_names[::-1])+"/"+getFileName(sample.filePath)+".fcs"
            self.__create_hdf5_file(sample = sample(), h5py_filehandle = h5py_filehandle, h5py_internal_name = h5py_internal_name)
            
        return None
    
    def __create_hdf5_file(self, sample, h5py_filehandle, h5py_internal_name):
        #sample is supposedly a pandas dataframe here
        if not isinstance(sample, pd.DataFrame):
            raise #Todo: better raise
        
        HDF5_metadata = pd.Series([h5py_filehandle,h5py_internal_name])
        
        sample.to_hdf(h5py_filehandle, key='fcs', mode='w')
        HDF5_metadata.to_hdf(h5py_filehandle, key='filenames', mode='a')
        
        return None