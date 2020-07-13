#!/usr/bin/env python3
#These flags/variables are accessible throughout aligater under namespace agconf
#example: agconf.minCells, agconf.ag_verbose
#They affect behaviour of several functions.

#If a parent population has less cells than this threshold, the resulting gated population will be considered invalid
minCells=200

#the cellFilter threshhold is the _initial_ required number of events per sample when it's first loaded
#If lower than this, the entire sample is skipped and strategy never applied
cellFilter=100000

#Truncate values below this treshold i.e. if any measured value is lower than this, the value is piled up at this level instead. Set with generous margin but might affect viewability of plots if too generous.
ag_trimMeasurements = -10000
#Similarly for max measurements (this is similar to flowjo > preferences > cytometer > custom log/linear scaling
ag_maxMeasurement = 214748

#Divider. Some machines outputs higher values and have a divider to downscale values for comparability. 
#Manufacturer typically provides these and most are already preset in software such as FlowJo
#In my experience, BD machines like FacsCanto and Fortessa have divider 1, BioRad machines like YETI and ZE5 have 10000
ag_Divider = 10000

#Sets verbosity level, if False only minor information and warnings are printed. Does not affect plotting behaviour.
ag_verbose=False

#AliGater output dir
ag_out="/media/ludvig/Project_Storage/BloodVariome/aligater_output/"

#AliGater tempdir (intermediate files such as downsampled images)
ag_tmp="/media/ludvig/Project_Storage/aligater_temp/"

#AliGater home
ag_home="/media/ludvig/Project_Storage/BloodVariome/aligater/"

#Check ag dirs are valid
from aligater.check_exec_mode import check_ag_dirs
check_ag_dirs(ag_out, ag_tmp, ag_home)

#This detects how aligater was started. 
#You can change which line is commented to set how aligater should behave. It mainly affects auto-plotting behaviour.
from aligater.check_exec_mode import type_of_script
execMode = type_of_script()
#Uncomment below line to force behaviour
#execMode = 'terminal'  #Options are 'terminal', 'ipython', 'jupyter'

#Below filters out FutureWarnings that tend to pop up now and then, usually from numpy, pandas or h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
