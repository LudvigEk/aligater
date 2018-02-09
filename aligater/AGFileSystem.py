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
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	http://nilssonlab.org

import aligater as ag
import numpy as np
import pandas as pd
import os
import sys

#TODO: Perhaps try to avoid conversion from DF -> nparray and back, performance hit?
def compensateDF(fcsDF, metaDict):
    spill_matrix=metaDict['SPILL'].split(',')
    n = int(spill_matrix[0]) #number of colors
    
    colNames=fcsDF.columns[4:(n+4)]
    fcsArray = fcsDF[colNames]
    
    comp_matrix = np.array(spill_matrix[n+1:]).reshape(n, n).astype(float)
    inv_comp_matrix = np.linalg.inv(comp_matrix)
    compensatedArray=np.dot(fcsArray, inv_comp_matrix)
    
    fcsDF.update(pd.DataFrame(compensatedArray,columns=colNames))
    
    return fcsDF

def loadFCS(path, compensate=True, metadata=False, return_type="index"):
    if not isinstance(return_type, str):
        raise TypeError("return_type must be specified as string and either of 'AGsample' or 'index'")
    if not return_type.lower() in ['agsample', 'index']:
        raise ValueError("return_type must be specified as string and either of 'AGsample' or 'index'")
    pardir=ag.getFileName(ag.getParent(path))
    parpardir=ag.getFileName(ag.getParent(ag.getParent(path)))
    sys.stderr.write("Opening file "+ag.getFileName(path)+" from folder /"+parpardir+"/"+pardir+"\n")
    metaDict,fcsDF = ag.parse(path,output_format='DataFrame')
    rows=fcsDF.shape[0]
    cols=fcsDF.columns[4:-1]
    sys.stderr.write("Loaded dataset with "+str(rows)+" events.\nMarker labels: ")
    for elem in cols:
        sys.stderr.write(elem+ " ")
    sys.stderr.write("\n")
    if compensate:
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
        if ".fcs" not in lFilter:
                lFilter.extend(".fsc")
    lOutput=[]
    for root, dirs, lFiles in os.walk(sSrc):
        for file in lFiles:
            filePath = os.path.join(root,file)
            lOutput.append(filePath)
            
    lFlaggedIndicies=[]
    for index, filePath in enumerate(lOutput):
        fileName=os.path.basename(filePath)  
        if lIgnoreTypes is not None:
            if any(ignoreType in os.path.splitext(fileName)[1] for ignoreType in lIgnoreTypes):
                lFlaggedIndicies.append(index)
                continue
        if lMask is not None:            
            if any(mask in fileName for mask in lMask): 
                lFlaggedIndicies.append(index)
                continue
        if lFilter is not None:             
            if any(sFilter not in fileName for sFilter in lFilter): 
                lFlaggedIndicies.append(index)
                continue    
    lOutput = [i for j, i in enumerate(lOutput) if j not in lFlaggedIndicies]
    sOutputString="Collected "+str(len(lOutput))+" files, "+str(len(lFlaggedIndicies))+" files did not pass filter(s) and mask(s).\n"
    sys.stderr.write(sOutputString)
    return pd.Series(lOutput)

def getFileName(sSrc):
    baseName=os.path.basename(sSrc)
    nameWithoutExtension=os.path.splitext(baseName)[0]
    return nameWithoutExtension

