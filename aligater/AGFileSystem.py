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
#	Utilizing Intels Math Kernel Library (MKL)
#
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	http://nilssonlab.org
import aligater as ag
import pandas as pd

sentinel = object()
def getGatedVector(fcsDF, gate, vI=sentinel):
    if vI is sentinel:
        vI=fcsDF.index
    gated_vector=fcsDF[gate].loc[vI]
    return gated_vector
