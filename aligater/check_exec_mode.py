import sys
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
