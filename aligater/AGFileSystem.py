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
import os
import sys
import copy #For deep copies of lists

#AliGater imports
from aligater.fscparser_api import parse
import aligater.AGConfig as agconf

#Dummy object for arguments
sentinel=object

#Some custom errors from template
class ExceptionTemplate(Exception):
    def __call__(self, *args):
        return self.__class__(*(args + self.args))
    def __str__(self):
        return ' '.join(self.args)


#General AliGater name to create errors
class AliGaterError(ExceptionTemplate): pass
#Some reoccuring, common ones
invalidAGgateError=AliGaterError('passed gate is not a valid AGgate object')
invalidAGgateParentError=AliGaterError('Invalid AGgate object passed as parentGate')
invalidSampleError=AliGaterError('Invalid AGsample object')
filePlotError=AliGaterError("If plotting to file is requested filePlot must be string filename")
markerError=AliGaterError("not present in sample, check spelling or control your dataframe.columns labels")


def compensateDF(fcsDF, metaDict, *args, **kwargs):
    #TODO spill_col_name in agconfig?
    if 'spill_col_name' in kwargs:
        spill_keyword = kwargs['spill_col_name']
    elif 'SPILL' in metaDict.keys():
        spill_keyword='SPILL'
    elif '$SPILL' in metaDict.keys():
        spill_keyword='$SPILL'
    elif 'SPILLOVER' in metaDict.keys():
        spill_keyword='SPILLOVER'
    elif '$SPILLOVER' in metaDict.keys():
        spill_keyword='$SPILLOVER'
    else:
        raise ValueError("Unknown label of spillover in metadata, pass correct alias with spill_col_name\nYou can load the fcs using ag.loadFCS with compensate=False, metadata=True and inspect the metadata for corect label.")
    
    spill_matrix=metaDict[spill_keyword].split(',')
    n = int(spill_matrix[0]) #number of colors
    
    colNames=fcsDF.columns[4:(n+4)]
    fcsArray = fcsDF[colNames]
    
    comp_matrix = np.array(spill_matrix[n+1:]).reshape(n, n).astype(float)
    #Sanity check that the compensation matrix is non-zero if compensation was requested
    tmp_identity = np.identity(n)
    if np.array_equal(comp_matrix, tmp_identity):
        reportStr="WARNING: No compensation data available in sample!\n"
        sys.stderr.write(reportStr)
    
    inv_comp_matrix = np.linalg.inv(comp_matrix)
    compensatedArray=np.dot(fcsArray, inv_comp_matrix)
    
    fcsDF.update(pd.DataFrame(compensatedArray,columns=colNames))
    
    return fcsDF

def reportGateResults(vI, vOutput):
    if agconf.ag_verbose:
        reportString="After gating, "+str(len(vOutput))+" out of "+str(len(vI))+" events remain.\n"
        sys.stderr.write(reportString)
    return

def compensate_manual(fcsDF, comp_matrix):
    n=len(comp_matrix)
    colNames=fcsDF.columns[4:(n+4)]
    fcsArray = fcsDF[colNames]
    #Sanity check that the compensation matrix is non-zero if compensation was requested
    #Raise here instead of warning?
    tmp_identity = np.identity(n)
    if np.array_equal(comp_matrix, tmp_identity):
        reportStr="WARNING: passed compensation matrix is 0, compensation NOT applied!\n"
        sys.stderr.write(reportStr)
        
    inv_comp_matrix = np.linalg.inv(comp_matrix)
    compensatedArray=np.dot(fcsArray, inv_comp_matrix)
    fcsDF.update(pd.DataFrame(compensatedArray,columns=colNames))
    reportStr="Successfully applied manual compensation\n"
    sys.stderr.write(reportStr)
    return fcsDF
    
def getCompensationMatrix(fcsDF, metaDict):
    spill_matrix=metaDict['SPILL'].split(',')
    n = int(spill_matrix[0]) #number of colors
    
    colNames=fcsDF.columns[4:(n+4)]
    
    comp_matrix = np.array(spill_matrix[n+1:]).reshape(n, n).astype(float)
    #Sanity check that the compensation matrix is non-zero if compensation was requested
    #Raise here instead of warning?
    tmp_identity = np.identity(n)
    if np.array_equal(comp_matrix, tmp_identity):
        reportStr="WARNING: This samples has no compensation data available\n"
        sys.stderr.write(reportStr)
    
    return colNames, comp_matrix

def getGatedVector(fcsDF, gate, vI=None, return_type="pdseries", dtype=np.float64):
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
    if vI is None:
        vI=fcsDF.index
    if return_type.lower()=="pdseries":
        gated_vector=fcsDF[gate].loc[vI]
    else:
        gated_vector=fcsDF[gate].loc[vI].values.astype(dtype)
    return gated_vector

def getGatedVectors(fcsDF, gate1, gate2, vI=None, return_type="pdseries"):
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
    if vI is None:
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


def loadFCS(path, compensate=True, metadata=False, comp_matrix=None, return_type="index", markers=sentinel, marker_names='label',ignore_minCell_filter=False):
    #********Lazy loading of*************
    #TODO; could move to AGClasses, kind of makes sense.
    from aligater.AGClasses import AGsample
    #************************************
    if not isinstance(return_type, str):
        raise TypeError("return_type must be specified as string and either of 'AGsample' or 'index'")
    if not return_type.lower() in ['agsample', 'index']:
        raise ValueError("return_type must be specified as string and either of 'AGsample' or 'index'")
    if not isinstance(marker_names,str):
        raise AliGaterError("in loadFCS:","invalid dtype in marker_names, expected "+str(type(str))+" found "+str(type(marker_names)))
    if not marker_names.lower() in ['label','color']:
        raise AliGaterError("in loadFCS:"," marker_names must be either of 'label' or 'color' found: "+str(marker_names))
    if markers is sentinel:
        checkMarkers=False
    else:
        checkMarkers=True
        if not isinstance(markers,(str, list)):
            raise ValueError("if markers is specified it must be list of str or single str")
    if comp_matrix is not None:
        if not isinstance(comp_matrix, np.ndarray):
            raise TypeError("Manual compensation matrix must be a numpy ndarray")
        
    pardir=getFileName(getParent(path))
    parpardir=getFileName(getParent(getParent(path)))
    sys.stderr.write("Opening file "+getFileName(path)+" from folder /"+parpardir+"/"+pardir+"\n")
    if marker_names.lower()=='label':
        channel_naming='$PnS'
    elif marker_names.lower()=='color':
        channel_naming='$PnN'

    metaDict,fcsDF = parse(path,output_format='DataFrame',channel_naming=channel_naming)
    rows=fcsDF.shape[0]
    cols=fcsDF.columns[4:-1]
    
    #SANITY CHECK: first four columns are some combination of foward and side scatters
    #scatter_cols=fcsDF.columns[0:4]
    #if not all([scatter in ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H'] for scatter in scatter_cols]):
    #    sys.stderr.write("First four columns of fcs file are not foward and side scatters, skippping\n")
    #    return None
    
    sys.stderr.write("Loaded dataset with "+str(rows)+" events.\n")
    if rows < agconf.cellFilter and not ignore_minCell_filter:
        if agconf.ag_verbose:
            sys.stderr.write("Sample has fewer events than cellFilter threshold, skipping\n")
        return None
    
    if agconf.ag_verbose:
        sys.stderr.write("Marker labels: ")
        for elem in cols:
            sys.stderr.write(elem+ " ")
        sys.stderr.write("\n")
    
    if checkMarkers:
        for elem in cols:
            if elem not in markers:
                reportStr=elem+" not in specified markers ("+str(markers)+"), skipping sample\n"
                sys.stderr.write(reportStr)
                return None
            
    if compensate:
        if comp_matrix is not None:
            sys.stderr.write("External compensation matrix passed, applying\n")
            fcsDF=compensate_manual(fcsDF, comp_matrix)
        else:
            fcsDF=compensateDF(fcsDF, metaDict)
    
    if not isinstance(agconf.ag_trimMeasurements, (float, int)):
        raise AliGaterError('in loadFCS: ','ag_trimMeasurements must be float or int, found: '+str(type(agconf.ag_trimMeasurements)))
    fcsDF=fcsDF.apply(lambda x: np.where(x < agconf.ag_trimMeasurements, agconf.ag_trimMeasurements, x))
    
    if metadata:
        if return_type.lower()=='agsample':
            return metaDict, AGsample(fcsDF,path)
        else:
            return metaDict, fcsDF
    else:
        if return_type.lower()=='agsample':
            return AGsample(fcsDF,path)
        else:
            return fcsDF

def getParent(sSrc):
    parentDir=os.path.abspath(os.path.join(sSrc, os.pardir))
    return parentDir

def collectFiles(sSrc, lFilter=None, lMask=None, lIgnoreTypes=None):
    if lIgnoreTypes is not None:
        if type(lIgnoreTypes) is not list:
            raise TypeError("lIgnoreTypes is not of type List, do lIgnoreTypes=['.FileEnding'] for single file ending strings")    
    if lMask is not None:
            if type(lMask) is not list:
                raise TypeError("lMask is not of type List, do lMask=['mask'] for single mask strings")
    if lFilter is not None:
        if type(lFilter) is not list:
            raise TypeError("lFilter is not of type List, do lFilter=['filter'] for single filter strings")
                
    lOutput=[]
    for root, dirs, lFiles in os.walk(sSrc):
        for file in lFiles:
            filePath = os.path.join(root,file)
            lOutput.append(filePath)
            
    lFlaggedIndicies=applyFilter(lOutput, lFilter,lMask, lIgnoreTypes)
    
    lOutput = [i for j, i in enumerate(lOutput) if j not in lFlaggedIndicies]
    sOutputString="Collected "+str(len(lOutput))+" files, "+str(len(lFlaggedIndicies))+" files did not pass filter(s) and mask(s).\n"
    sys.stderr.write(sOutputString)
    return pd.Series(lOutput)

def check_exists(sSrc):
    if not isinstance(sSrc, list):
        raise AliGaterError("in check_exists: ","invalid input dtype, expected "+str(type(list))+"found: "+str(type(sSrc)))
    if not all(isinstance(x, str) for x in sSrc):
       raise AliGaterError("in check_exists: ","invalid dtype in input list, all elements must be str")
    invalid_paths=[]
    not_fcs=[]
    for item in sSrc:
        if not os.path.isfile(item):
            invalid_paths.append(item)
        else:
            if not len(item) > 4:
                raise
            file_ending = item[-4:]
            if not file_ending == ".fcs":
                not_fcs.append(item)
    if len(invalid_paths) == 0:
        sys.stderr.write("All file paths exists.\n")
    else:
        reportStr = str(len(invalid_paths))+" file(s) were invalid/does not exists, note that linux file paths are case sensitive\nInvalid paths:\n"
        sys.stderr.write(reportStr)
        for elem in invalid_paths:
            sys.stderr.write(elem)
        raise AliGaterError("in check_exists: ","All file paths must be valid")
        
    if len(not_fcs) != 0:
        reportStr="Some file(s) did not have .fcs file endings and will be skipped: \n"
        sys.stderr.write(reportStr)
        for elem in not_fcs:
            sys.stderr.write(elem)
    result=copy.deepcopy(sSrc)
    for i in np.arange(len(result)-1,-1,-1):
        if (result[i] in invalid_paths) or (result[i] in not_fcs):
            del result[i]
    return result
    
def applyFilter(lOutput, lFilter, lMask, lIgnoreTypes):
    lFlaggedIndicies=[]
    for index, filePath in enumerate(lOutput):
        fileName=os.path.basename(filePath)
        if '.fcs' not in fileName:
            lFlaggedIndicies.append(index)
            continue
        if lIgnoreTypes is not None:
            if any(ignoreType in os.path.splitext(fileName)[1] for ignoreType in lIgnoreTypes):
                lFlaggedIndicies.append(index)
                continue
        if lMask is not None:            
            if any(mask in filePath for mask in lMask): 
                lFlaggedIndicies.append(index)
                continue
        #OBS, Change on 09-04-2018
        if lFilter is not None:             
            if all(sFilter not in filePath for sFilter in lFilter): 
                lFlaggedIndicies.append(index)
                continue 
    return lFlaggedIndicies

def listDir(dirList):
    #Recursive function that lists subfolders
    if isinstance(dirList,str):
        nextLevel = [os.path.join(dirList,i) for i in next(os.walk(dirList))[1]]
        dirList=nextLevel
    else:
        nextLevel=[]
        for elem in dirList:
            nextLevel.extend([os.path.join(elem,i) for i in next(os.walk(elem))[1]])
            
    if len(nextLevel)==0:
        return dirList
    else:
        dirList.extend(listDir(nextLevel))
        return list(set(dirList))

def getFileName(sSrc):
    baseName=os.path.basename(sSrc)
    nameWithoutExtension=os.path.splitext(baseName)[0]
    return nameWithoutExtension

