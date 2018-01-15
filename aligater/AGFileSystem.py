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
import pandas as pd
import os

sentinel = object()
def getGatedVector(fcsDF, gate, vI=sentinel):
    if vI is sentinel:
        vI=fcsDF.index
    gated_vector=fcsDF[gate].loc[vI]
    return gated_vector

def getGatedVectors(fcsDF, gate1, gate2, vI=sentinel):
    if vI is sentinel:
        vI=fcsDF.index
    gated_vector1=fcsDF[gate1].loc[vI]
    gated_vector2=fcsDF[gate2].loc[vI]
    return gated_vector1, gated_vector2

def getParent(sSrc):
    parentDir=os.path.abspath(os.path.join(sSrc, os.pardir))
    return parentDir

def collectFiles(sSrc, lFilter=None, lMask=None, lIgnoreTypes=None):
    lOutput=[]
    for root, dirs, lFiles in os.walk(sSrc):
        for file in lFiles:
            filePath = os.path.join(root,file)
            lOutput.append(filePath)
            
    lFlaggedIndicies=[]
    for index, filePath in enumerate(lOutput):
        fileName=os.path.basename(filePath)
        if lIgnoreTypes is not None:
            if type(lIgnoreTypes) is not list:
                raise TypeError("lIgnoreTypes is not of type List, do lIgnoreTypes=['.FileEnding'] for single file ending strings")
            if any(ignoreType in os.path.splitext(fileName)[1] for ignoreType in lIgnoreTypes):
                lFlaggedIndicies.append(index)
                continue
            
        if lMask is not None:
            if type(lMask) is not list:
                raise TypeError("lMask is not of type List, do lMask=['mask'] for single mask strings")
            if any(mask in fileName for mask in lMask): 
                lFlaggedIndicies.append(index)
                continue 
            
        if lFilter is not None:
            if type(lFilter) is not list:
                raise TypeError("lFilter is not of type List, do lFilter=['filter'] for single filter strings")
            if any(sFilter not in fileName for sFilter in lFilter): 
                lFlaggedIndicies.append(index)
                continue    
            
    lOutput = [i for j, i in enumerate(lOutput) if j not in lFlaggedIndicies]
    return pd.Series(lOutput)

def getFileName(sSrc):
    baseName=os.path.basename(sSrc)
    nameWithoutExtension=os.path.splitext(baseName)[0]
    return nameWithoutExtension

