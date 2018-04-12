from aligater.AGConfig import *
from aligater.AGFileSystem import *
from aligater.AGPlotRoutines import *
from aligater.AGCore import *
from aligater.AGClasses import *
from aligater.AGExperimental import *
from aligater.fscparser_api import parse
from aligater.AGCython import *
ag.execMode=ag.type_of_script()
reportStr="AliGater started in "+ag.execMode+" mode\n"
sys.stderr.write(reportStr)
