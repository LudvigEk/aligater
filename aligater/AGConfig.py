#!/usr/bin/env python3
#These flags/variables are accessible throughout aligater under namespace agconf
#example: agconf.minCells, agconf.ag_verbose

#Make sure to review and set up these filters to reasonable values for your flow cytometry setup

#If a population has less cells than this threshold, any population defined with this population as parent will be flagged invalid
minCells=300

#The cellFilter threshold is the _initial_ required number of events per sample and is checked when the sample is loaded
#If lower than this, the entire sample is skipped and strategy never applied
cellFilter=100000

#Truncate intensities below this treshold i.e. if any measured value is lower than this, the value is piled up at this level instead. 
#Set with generous margin but might affect viewability of plots if too generous.
ag_trimMeasurements = -10000

#Similarly for max measurements (similar settings to these can be found in flowjo > preferences > cytometer > custom log/linear scaling
ag_maxMeasurement = 214000

#Sets verbosity level, if False only minor information and warnings are printed. Does not affect plotting behaviour.
ag_verbose=False

import os.path
ag_home = os.path.dirname(os.path.dirname(__file__)) #AliGater Root folder, wherever it was

#AliGater output dir, change if needed
ag_out=os.path.join(ag_home, "out")

#AliGater tempdir/scratch space (intermediate files such as downsampled images), change if needed
ag_tmp=os.path.join(ag_home, "temp")



#Check ag dirs are valid
from aligater.check_exec_mode import check_agconfig #CHECK THAT THIS SANITIZES PATH
check_agconfig(ag_out, ag_tmp, ag_home, ag_verbose, minCells, cellFilter, ag_trimMeasurements, ag_maxMeasurement)

#This detects how aligater was started. 
#You can change which line is commented to set how aligater should behave. It mainly affects auto-plotting behaviour.
from aligater.check_exec_mode import type_of_script
execMode = type_of_script()

#Uncomment below line to force behaviour
#execMode = 'terminal'  #Options are 'terminal', 'ipython', 'jupyter'

#Below filters out FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)