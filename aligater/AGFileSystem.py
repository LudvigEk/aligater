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
#   Distributed under the MIT License

import numpy as np
import pandas as pd
import os
import sys
import copy #For deep copies of lists
import h5py
import json #for importing json with machine-specific scaling factors

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


def compensateDF(fcsDF, metaDict, fsc_ssc_count, *args, **kwargs):

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
    if not (n == len(fcsDF.columns)-fsc_ssc_count):
        raise AliGaterError("in LoadFCS, compensateDF: ", "unexpected number of channels. If your FCS are exported with Area+width+height values for each flourochrome the flourochrome_area_filter needs to be set to True.")
    #Depending on version FCS different number of cols-per-flourochrome will be reported(A-W-H vs just A) , and different number of columns preceding and subsequent the flourchrome columns
    #NOTE: AliGater assumes first columns are FSC/SSC
    colNames=fcsDF.columns[fsc_ssc_count:(n+fsc_ssc_count)]
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

def compensate_manual(fcsDF, comp_matrix, fsc_ssc_count):
    n=len(comp_matrix)
    if not (n == len(fcsDF.columns)-fsc_ssc_count):
        raise AliGaterError("in LoadFCS, compensate_manual: ", "unexpected number of channels. If your FCS are exported with Area+width+height values for each flourochrome the flourochrome_area_filter needs to be set to True.")
    #Depending on version FCS different number of cols-per-flourochrome will be reported(A-W-H vs just A) , and different number of columns preceding and subsequent the flourchrome columns
    #NOTE: AliGater assumes first columns are FSC/SSC
    colNames=fcsDF.columns[fsc_ssc_count:(n+fsc_ssc_count)]
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
    reportStr="Applied passed compensation matrix\n"
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

def loadHDF5sample(path, sampling_resolution=32):
    """
    Loads a HDF5 compressed fcs file, these are created with aligater.AGClasses.AGExperiment.create_HDF5_files
    This function is mainly intended for internal use in the aligater.AGClasses.AGExperiment.apply function.
    
    **Parameters**
    
    path : str
        Absolute path to .fcs file.
    
    sampling_resolution : int, default: 32
        To-be-deprecated.
        Used to specify downsampling dimensions for batch runs through aligater.AGClasses.AGExperiments, this parameter is just passed through.\n
        Will be moved into a ``*args`` ``**kwargs`` style parameter.\n
        Should always be ignored if loading single files - it does nothing.
    
    **Returns**
    
    aligater.AGClasses.AGSample
    
    """
    #********Lazy loading of*************
    # could move to AGClasses and avoid, kind of makes sense.
    from aligater.AGClasses import AGsample
    #************************************
    
    fcsDF = pd.read_hdf(path, key='fcs')
    metadata = pd.read_hdf(path, key='filenames')
    h5py_internal_name = metadata.iloc[1]
    
    reportStr="Opening "+str(metadata[0])+" which is compressed from "+str(metadata[1])+"\n"
    sys.stderr.write(reportStr)
    rows = len(fcsDF)
    if agconf.ag_verbose:
        sys.stderr.write("Loaded dataset with "+str(rows)+" events.\n")
    return AGsample(fcsDF, h5py_internal_name, sampling_resolution=sampling_resolution)


def loadFCS(path, compensate=True, metadata=False, comp_matrix=None, return_type="index", markers=None, marker_names='label',ignore_minCell_filter=False, flourochrome_area_filter=False, sampling_resolution=32, nOfEvents=None):
    """
    Loads an fcs file from given path into an aligater.AGClasses.AGSample object. \n
    Multiple settings dealing with compensation, checking for markers etc.
    
    **Parameters**
    
    path : str
        Absolute path to .fcs file.
    compensate : bool, default: True
        Apply compensation based on the compensation matrix in file metadata.
    comp_matrix : np.ndarray or None, default: None
        Optional passed external compensation matrix to apply for compensation.
    return_type : str, optional, default: "index"
        Specifies if the loaded flow intensities should be return as an aligater.AGClasses.AGSample object ("agsample") or a pandas.Dataframe ("index").
    markers : list-like, optional, default: None
        A list-like containing str names of the expected markers in the sample. Loading will fail if any marker is missing and None is instead returned.
    marker_names : str, default: 'label'
        Options are 'label' or 'color'\n
        Specify which labelling style should be used for the flow data - the flourochrome marker or the labels, both are read from the .fcs files metadata.
    ignore_minCell_filter : bool, default: False
        If true, ignores the minCell filter for total sample events specified in aligater.AGConf.minCells and loads the sample anyway.
    flourochrome_area_filter : bool, default : False
        Some machines/fcs files report Area, width and height for each flourochrome, some only report the area signal.\n
        This flag enables filtering of the height and width channels for flourochromes and only return the area.\n
        For side- and forward-scatter channels all three are always returned.
    sampling_resolution : int, default: 32
        To-be-deprecated.
        Used to specify downsampling dimensions for batch runs through aligater.AGClasses.AGExperiments, this parameter is just passed through.\n
        Will be moved into a ``*args`` ``**kwargs`` style parameter.\n
        Should always be ignored if loading single files - it does nothing.
    
    **Returns**
    
    dict (optional)
        if metadata is True a dictionary object containing metadata information is returned first, in addition to the flow data.
    pandas.core.dataframe
        If return type is 'index'
    aligater.AGClasses.AGSample
        If return_type is 'agsample' returns an AGSample object with the flow data loaded
        
    .. note::   
        If any checks fail such as marker names were supplied and do not match marker names in the sample or the sample containing fewer total events than aligater.AGConf.minCell, None is instead returned.
        
    **Examples**

    None currently.
    """
    #********Lazy loading of*************
    # could move to AGClasses and avoid, kind of makes sense.
    from aligater.AGClasses import AGsample
    #************************************
    if not isinstance(return_type, str):
        raise TypeError("return_type must be specified as string and either of 'AGsample' or 'index'")
    if not return_type.lower() in ['agsample', 'index']:
        raise ValueError("return_type must be specified as string and either of 'AGsample' or 'index'")
    if not isinstance(marker_names,str):
        raise AliGaterError("in loadFCS:","invalid dtype in marker_names, expected "+str(type(str))+" found "+str(type(marker_names)))
    if not marker_names.lower() in ['label','color']:
        raise AliGaterError("in loadFCS:","marker_names must be either of 'label' or 'color' found: "+str(marker_names))
    if nOfEvents is not None:
        if not isinstance(nOfEvents, int):
            raise AliGaterError("in loadFCS:","nOfEvents must be integer or None, found: "+str(type(nOfEvents)))
    if markers is None:
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

    metaDict,fcsDF = parse(path,output_format='DataFrame',channel_naming=channel_naming, nOfEvents=nOfEvents) #add optional n of events to read
    rows=fcsDF.shape[0] # n of events
    if int(metaDict['$TOT']) != int(rows) and nOfEvents is not None:
        if agconf.ag_verbose:
            #Trimmed file
            reportStr="Trimmed sample to "+str(nOfEvents)+"/"+str(metaDict["$TOT"])+" events.\n"
            sys.stderr.write(reportStr)
    elif int(metaDict['$TOT']) != int(rows):
        raise AliGaterError("FCS file has fewer events than specified in metadata, possibly corrupt.")
        
    #Instead: print warning of how many events skipped & what was total
    
    #Update
    #Depending on choices by the operator on export, machine, and FCS standard there can be different number of preceding and subsequent columns from the actual flourochrome cols
    #Also each flourochrome can be exported with Area, Width, Height or only with Area
    #This part attempts to parse this
    #AliGater will attempt to only extract Area for the flourochromes while keeping height width area for for scatters.
    #It also tries to drop redundant columns that sometimes exists such as Time, TLSW, TMSW, Event info
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
            if col.lower()[-2:] != '-a' and not bExclusion and flourochrome_area_filter:
                indicies_to_drop.append(index)
                continue
        else:
            fsc_ssc_count += 1
    fcsDF.drop(fcsDF.columns[indicies_to_drop], axis=1, inplace=True)
    cols = fcsDF.columns
    #cols=fcsDF.columns[4:-1]
    
    #SANITY CHECK: first four columns are some combination of foward and side scatters
    #scatter_cols=fcsDF.columns[0:4]
    #if not all([scatter in ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H'] for scatter in scatter_cols]):
    #    sys.stderr.write("First four columns of fcs file are not foward and side scatters, skippping\n")
    #    return None
    if agconf.ag_verbose:
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
            fcsDF=compensate_manual(fcsDF, comp_matrix, fsc_ssc_count)
        else:
            fcsDF=compensateDF(fcsDF, metaDict, fsc_ssc_count)
    
    if not isinstance(agconf.ag_trimMeasurements, (float, int)):
        raise AliGaterError('in loadFCS: ','ag_trimMeasurements must be float or int, found: '+str(type(agconf.ag_trimMeasurements)))
    if not isinstance(agconf.ag_maxMeasurement, (float, int)):
        raise AliGaterError('in loadFCS: ','ag_maxMeasurement must be float or int, found: '+str(type(agconf.ag_maxMeasurement)))
        
    #adding support for json instead of ag_divider
    with open(agconf.ag_home+"/aligater/AGMachineDividers.json",'r') as json_filehandle:
        divider_dict = json.load(json_filehandle)
    #Loop over machines in divider_dict, check that 
    if metaDict['$CYT'] in divider_dict.keys():
        ag_Divider = int(divider_dict[metaDict['$CYT']])
    else:
        reportStr="WARNING: in loadFCS: unknown machine "+str(metaDict['$CYT'])+" encounter in sample, scaling factor might be wrong.\nPlease add it manually to AGMachineDividers.json\n"
        sys.stderr.write(reportStr)
        ag_Divider=1
    #if not isinstance(agconf.ag_Divider, (int)):
    #    raise AliGaterError('in loadFCS: ','ag_Divider must be int, found: '+str(type(agconf.ag_Divider)))
        
    #Apply divider before pileup
    fcsDF=fcsDF.apply(lambda x: x/ag_Divider)
    #Apply lower truncation
    fcsDF=fcsDF.apply(lambda x: np.where(x < agconf.ag_trimMeasurements, agconf.ag_trimMeasurements, x))
    #Apply upper truncation
    fcsDF=fcsDF.apply(lambda x: np.where(x > agconf.ag_maxMeasurement, agconf.ag_maxMeasurement, x))
    
    if metadata:
        if return_type.lower()=='agsample':
            return metaDict, AGsample(fcsDF,path, sampling_resolution=sampling_resolution)
        else:
            return metaDict, fcsDF
    else:
        if return_type.lower()=='agsample':
            return AGsample(fcsDF,path, sampling_resolution=sampling_resolution)
        else:
            return fcsDF

def getParent(sSrc):
    parentDir=os.path.abspath(os.path.join(sSrc, os.pardir))
    return parentDir

def collectFiles(sSrc, lFilter=None, lMask=None, lIgnoreTypes=None, HDF5=False):
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
            
    lFlaggedIndicies=applyFilter(lOutput, lFilter,lMask, lIgnoreTypes, HDF5)
    
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
    
def applyFilter(lOutput, lFilter, lMask, lIgnoreTypes, HDF5=False):
    lFlaggedIndicies=[]
    for index, filePath in enumerate(lOutput):
        fileName=os.path.basename(filePath)
        if HDF5:
            if '.h5' not in fileName:
                lFlaggedIndicies.append(index)
                continue
        else:
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

def __loadH5FCS(path, sampling_resolution=32):
    #********Lazy loading of*************
    # could move to AGClasses and avoid, kind of makes sense.
    from aligater.AGClasses import AGsample
    #************************************    
    with h5py.File(path,'r') as hf:
            sa2 = hf['intensities'][:]
            filePath = hf['filePath']
    fcsDF = pd.DataFrame(sa2)
    fcs = AGsample(fcsDF, filePath, sampling_resolution=sampling_resolution)
    return fcs