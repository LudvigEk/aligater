import sys
import os
def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            reportStr="AliGater started in Jupyter mode\n"
            sys.stderr.write(reportStr)
            return 'jupyter'
        if 'terminal' in ipy_str:
            reportStr="AliGater started in ipython mode\n"
            sys.stderr.write(reportStr)
            return 'ipython'
    except:
        #If not available, default to terminal
        reportStr="AliGater started in terminal mode\n"
        sys.stderr.write(reportStr)
        return 'terminal'

def check_agconfig(ag_out, ag_tmp, ag_home, ag_verbose, minCells, cellFilter, ag_trimMeasurements, ag_maxMeasurement):
    if not check_filePath(ag_home) == "dir":
        raise NameError("ag_home not correctly set (this is a bug)")
    if not check_filePath(ag_tmp) == "dir":
        raise NameError("ag_tmp must be string filepath to existing directory, please review your AGConfig.py")
    if not check_filePath(ag_out) == "dir":
        raise NameError("ag_out must be string filepath to existing directory, please review your AGConfig.py")
    if not isinstance(ag_verbose,bool):
        raise TypeError("ag_verbose must be bool (False/True), found: "+str(type(ag_verbose))+", please review your AGConfig.py")
    scalar_filters=[minCells, cellFilter, ag_trimMeasurements, ag_maxMeasurement]
    filter_names=['minCells','cellFilter','ag_trimMeasurements','ag_maxMeasurement']
    for i in range(0,len(scalar_filters)):
        if not isinstance(scalar_filters[i], int):
            reportStr=filter_names[i]+" must be int, found: "+str(type(scalar_filters[i]))+", please review your AGConfig.py"
            raise TypeError(reportStr)

	
def check_filePath(filepath):
    if not isinstance(filepath,str):
        raise TypeError("Filepath must be string, found "+str(type(filepath)))
    if os.path.exists(filepath):
        if os.path.isfile(filepath):
            return "file"
        elif os.path.isdir(filepath):
            return "dir"
    else:
        #If the filepath is a string, but doesn't exists,
        #check if directory to that filepath exists
        dirname = os.path.dirname(filepath)
        if os.path.exists(dirname):
            return "dir_exists"
    return False

