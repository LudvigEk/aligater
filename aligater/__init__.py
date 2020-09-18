#Define functions that users should have access to here
#For dev purposes all are accessible currently. 

#First four python loads must be in correct order to avoid circular loading. Basic functions and the aligater classes. 
from aligater.AGConfig import execMode,ag_verbose,minCells,cellFilter
#Wrapper for AGCython import with a simple reminder on py3.6+ ipykernel while using ipython/Jupyter and py3.6+ otherwise. 
#If that's not the reason for your error I apologize sincerely for misleading.
import sys

try:
	from aligater.AGCythonUtils import *
except ImportError as e:
	if execMode in ['jupyter','ipython']:
		sys.stderr.write("Couldn't import AGCythonUtils module, error: {0}\nDo you have an ipykernel with python 3.6+ active?\n".format(e))
	else:
		sys.stderr.write("Couldn't import AGCythonUtils, error: {0}\nDo you have a python 3.6+ environment active?\n".format(e))
	raise(e)

from aligater.AGFileSystem import *
from aligater.AGPlotRoutines import *
from aligater.AGClasses import *

#Another Cython import wrapper
try:
	from aligater.AGCython import *
except ImportError as e:
	if execMode in ['jupyter','ipython']:
		sys.stderr.write("Couldn't import cython module, error: {0}\nDo you have an ipykernel with python 3.6+ active?\n".format(e))
	else:
		sys.stderr.write("Couldn't import cython module, error: {0}\nDo you have a python 3.6+ environment active?\n".format(e))
	raise(e)

from aligater.AGCore import *
from aligater.AGExperimental import *
