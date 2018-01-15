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
import numpy as np

#TODO
def compensateDF(fcsDF, metaDict):
    m0=fcsDF.as_matrix()
    spill_matrix=metaDict['SPILL'].split(',')
    n = int(spill_matrix[0])
    colors = spill_matrix[1:n+1]
    comp_matrix = np.array(spill_matrix[n+1:]).reshape(n, n)
    compensatedArray=np.dot(m0, comp_matrix)
    return pd.DataFrame(compensatedArray)

def someLoadFunction(path, compensate=False, metadata=False, separator='\t'):
    metaDict,fcsDF = ag.parse(path,output_format='DataFrame')
    if compensate:
        fcsDF=compensateDF(fcsDF, metaDict)
    if metadata:
        return metaDict, fcsDF
    else:
        return fcsDF