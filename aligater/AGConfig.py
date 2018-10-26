#!/usr/bin/env python3
#These flags/variables are accessible throughout aligater under namespace agconf
#example: agconf.minCells, agconf.ag_verbose
#They affect behaviour of several functions.

#If a parent population has less cells than this threshold, the resulting gated population will be considered invalid
minCells=300


#the cellFilter threshhold is the _initial_ required number of events per sample when it's first loaded
#If lower than this, the entire sample is skipped and strategy never applied
cellFilter=100000

#Sets verbosity level, if False only minor information and warnings are printed. Does not affect plotting behaviour.
ag_verbose=False

#AliGater output dir
ag_out="/media/ludvig/Project_Storage/BloodVariome/aligater_output/"

#AliGater tempdir (intermediate images and downsampled images)
ag_tmp="/media/ludvig/Project_Storage/aligater_temp/"

#AliGater home
ag_home="/media/ludvig/Project_Storage/BloodVariome/aligater/"

#This detects how aligater was started. 
#You can change which line is commented to set how aligater should behave. It mainly affects auto-plotting behaviour.
from aligater.check_exec_mode import type_of_script
execMode = type_of_script()
#Uncomment below line to force behaviour
#execMode = 'terminal'  #Options are 'terminal', 'ipython', 'jupyter'

from aligater.check_exec_mode import check_ag_dirs
check_ag_dirs(ag_out, ag_tmp, ag_home)
