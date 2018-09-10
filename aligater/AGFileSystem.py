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
import pandas as pd
import os
import sys
sentinel=object
#TODO: Perhaps try to avoid conversion from DF -> nparray and back, performance hit?
def compensateDF(fcsDF, metaDict):
    spill_matrix=metaDict['SPILL'].split(',')
    n = int(spill_matrix[0]) #number of colors
    
    colNames=fcsDF.columns[4:(n+4)]
    fcsArray = fcsDF[colNames]
    
    comp_matrix = np.array(spill_matrix[n+1:]).reshape(n, n).astype(float)
    #Sanity check that the compensation matrix is non-zero if compensation was requested
    #Raise here instead of warning?
    tmp_identity = np.identity(n)
    if np.array_equal(comp_matrix, tmp_identity):
        reportStr="WARNING: No compensation data available, compensation NOT applied!\n"
        sys.stderr.write(reportStr)
    
    inv_comp_matrix = np.linalg.inv(comp_matrix)
    compensatedArray=np.dot(fcsArray, inv_comp_matrix)
    
    fcsDF.update(pd.DataFrame(compensatedArray,columns=colNames))
    
    return fcsDF

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
        reportStr="WARNING: No compensation data available, compensation NOT applied!\n"
        sys.stderr.write(reportStr)
    
    return colNames, comp_matrix
    
def loadFCS(path, compensate=True, metadata=False, comp_matrix=None, return_type="index", markers=sentinel):
    if not isinstance(return_type, str):
        raise TypeError("return_type must be specified as string and either of 'AGsample' or 'index'")
    if not return_type.lower() in ['agsample', 'index']:
        raise ValueError("return_type must be specified as string and either of 'AGsample' or 'index'")
    if markers is sentinel:
        checkMarkers=False
    else:
        checkMarkers=True
        if not isinstance(markers,(str, list)):
            raise ValueError("if markers is specified it must be list of str or single str")
    if comp_matrix is not None:
        if not isinstance(comp_matrix, np.ndarray):
            raise TypeError("Manual compensation matrix must be a numpy ndarray")
    pardir=ag.getFileName(ag.getParent(path))
    parpardir=ag.getFileName(ag.getParent(ag.getParent(path)))
    if ag.ag_verbose:
        sys.stderr.write("Opening file "+ag.getFileName(path)+" from folder /"+parpardir+"/"+pardir+"\n")
    metaDict,fcsDF = ag.parse(path,output_format='DataFrame')
    rows=fcsDF.shape[0]
    cols=fcsDF.columns[4:-1]
    
    #SANITY CHECK: first four columns are some combination of foward and side scatters
    scatter_cols=fcsDF.columns[0:4]
    if not all([scatter in ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H'] for scatter in scatter_cols]):
        sys.stderr.write("First four columns of fcs file are not foward and side scatters, skippping\n")
        return None
        
    if ag.ag_verbose:
        sys.stderr.write("Loaded dataset with "+str(rows)+" events.\n")
    if rows < ag.cellFilter:
        if ag.ag_verbose:
            sys.stderr.write("Sample has fewer events than cellFilter threshold, skipping\n")
        return None
    
    if ag.ag_verbose:
        sys.stderr.write("Marker labels: ")
        for elem in cols:
            sys.stderr.write(elem+ " ")
        sys.stderr.write("\n")
    
    if checkMarkers:
        for elem in cols:
            if elem not in markers:
                reportStr=elem+" not in specified markers, skipping sample\n"
                sys.stderr.write(reportStr)
                return None
            
    if compensate:
        if comp_matrix is not None:
            fcsDF=compensate_manual(fcsDF, comp_matrix)
        else:
            fcsDF=compensateDF(fcsDF, metaDict)
            
    if metadata:
        if return_type.lower()=='agsample':
            return metaDict, ag.AGsample(fcsDF,path)
        else:
            return metaDict, fcsDF
    else:
        if return_type.lower()=='agsample':
            return ag.AGsample(fcsDF,path)
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

