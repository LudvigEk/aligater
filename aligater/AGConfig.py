#!/usr/bin/env python3
#These flags are accessible throughout aligater under it's namespace, i.e. ag.minCells, ag.cellFilter, ag.ag_verbose.
#They affect behaviour of several functions.

#the minCells threshold decides a minimum level below which a population is considered non-existent. 
#I.e. below this threshold it's essentially 'rounded' to zero
minCells=5

#the cellFilter threshhold is the _initial_ required number of events per sample when it's first loaded
#If lower than this, the entire sample is skipped and strategy is not applied
cellFilter=100000

#Sets verbosity level, if False only warnings are printed. Does not affect plotting.
ag_verbose=True
