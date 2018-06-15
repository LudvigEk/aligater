#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#	______|\__________\o/__________
#			~~aliGater~~
#	(semi)automated gating software
#
#               /^^\
#   /^^\_______/0  \_
#  (                 `~+++,,_________,,++~^^^^^^^
#..V^V^V^V^V^V^\.................................
#
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	http://nilssonlab.org

import numpy as np
from matplotlib.ticker import Locator, Formatter
import aligater as ag
from matplotlib import rcParams
import pandas as pd
import sys, os, errno
import gc   #for manually calling garbage collection during large iterations
import six
            
sentinel=object()   

#Imported StackOverflow progressbar method. Credit to user Brian Khuu.
#https://stackoverflow.com/questions/3160699/python-progress-bar/15860757#15860757
def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), format(progress*100,'.2f'), status)
    sys.stderr.write(text)
    sys.stderr.flush()


class AGgate:
    current=[]
    parent=[]
    name=None
    parentName=None
    
    def __init__(self, vI, vIParent, name, parentName):
        self.current=vI
        self.parent=vIParent
        self.name=name
        self.parentName=parentName
        
    def __call__(self):
        if not list(self.current):
            sys.stderr.write("This AGGate object does not contain any data\n")
            return []
        return self.current
    
    def changeParent(self, vIParent):
        self.parent=vIParent
        #THIS A GOOD IDEA?
        return self
    
    def getParent(self):
        if not list(self.parent):
            sys.stderr.write("This AGGate object does not contain any parent data\n")
            return []
        return self.parent
    
    def reportStats(self):
        if not list(self.current):
            raise ValueError("This AGGate object does not contain any data")
        elif not list(self.parent):
            sys.stderr.write("This AGGate object does not have any parent, only reporting current\n")
            return str(len(self.current))
        else:
            nOfcurrent = float(len(self.current))
            nOfparent = float(len(self.parent))
            if any([nOfcurrent==0, nOfparent==0]):
                sys.stderr.write("this AGGate object have reached 0 population (no gated events)\n")
                return str(nOfparent)+"\t"+str(nOfcurrent)+"\t0"
            outputStr=str(nOfparent)+"\t"+str(nOfcurrent)+"\t"+str(nOfcurrent/nOfparent)
            return outputStr
    
#    def downSample(self, fcsDF, limits):
#        ag.getHeatmap()
    

class AGsample:
    #vGate intended as list of tuple(name, gate)
    vGates=None
    sample=None
    filePath=None
    fcsDF=None
    gate=None
    downSample=None
    
    def __init__(self, fcsDF, filePath, sampleName = sentinel):
        if not isinstance(fcsDF, pd.DataFrame):
            raise
        if not isinstance(filePath, str):
            raise
        if sampleName is not sentinel:
            if not isinstance(sampleName, str):
                raise
        else:
            #If no name specified, set it to the filename without extension
            datename=ag.getFileName(ag.getParent(ag.getParent(filePath)))
            platename=ag.getFileName(ag.getParent(filePath))
            sampleName=datename+"/"+platename+"/"+ag.getFileName(filePath)
            
        self.fcsDF=fcsDF
        self.sample=sampleName
        self.vGates=[]
        self.filePath=filePath
    
    def __call__(self, name=sentinel):
        if name is sentinel:
            if not isinstance(self.fcsDF, pd.DataFrame):
                sys.stderr.write("sample does not contain any data\n")
                return None
            return self.fcsDF
        else:
            if not isinstance(name, str):
                raise
            else:
                if not self.vGates:
                    sys.stderr.write("sample does not contain any gates\n")
                    return None
                for index, gate in enumerate(self.vGates):
                    if gate.name==name:
                        return gate()
                sys.stderr.write(name+" not in sample name list\n")
                return None
        
    def update(self, gate, QC=False):
        #print(type(gate))
        #print(type(ag.AGClasses.AGgate))
        if not isinstance(gate,ag.AGClasses.AGgate):
            raise TypeError("gate is not a valid AGgate object")
        self.vGates.append(gate)
        if QC:
            self.downSample = self.downSample(gate, 16)
    
    def downSample(self,gate, bins):
        vX, vY = ag.getGatedVectors(self.fcsDF, "FSC-A", "SSC-A", vI=gate.parent, return_type="nparray")
        flaggedIndicies=[]
        for i in np.arange(0,len(vX),1):
            if vX[i]<0 or vX[i] > 500000 or vY[i] < 0 or vY[i] > 500000:
                flaggedIndicies.append(i)
        vX=np.delete(vX, flaggedIndicies)
        vY=np.delete(vY, flaggedIndicies)
        result = ag.getHeatmap(vX, vY, bins, "linear", "linear", "linear", normalize=True)
        return result[0]
        
    def full_index(self):
        return list(self.fcsDF.index.values)
    
    def report(self):
        reportStr="Gating report for "+self.sample+" ("+str(len(self.fcsDF.index))+" total events)\n"+str(len(self.vGates))+" gate(s):\n"
        for gate in self.vGates:
            reportStr=reportStr+str(gate.name)+" ("+str(len(gate()))+" events)\n"
        print(reportStr)
        
    def printData(self, file, precision=4, header=True):
        if header:
            #header
            header="date\tplate\tSample\tTotal"
            for gate in self.vGates:
                header=header+"\t"+str(gate.name)+"\t"+str(gate.name)+"/"+str(gate.parentName)
            header=header+"\n"
            file.write(header)
        #Data
        reportStr=ag.getFileName(ag.getParent(ag.getParent(self.filePath)))+"\t"+ag.getFileName(ag.getParent(self.filePath))+"\t"+self.sample+"\t"+str(len(self.full_index()))
        precision="."+str(precision)+"f"
        for gate in self.vGates:
            if len(gate[1].getParent()) == 0 or len(gate[1]()) == 0:
                fraction=str(0)
            else:
                fraction=str(format(len(gate[1]())/len(gate[1].getParent()),precision))
            reportStr=reportStr+"\t"+str(len(gate[1]()))+"\t"+fraction
        reportStr=reportStr+"\n"
        file.write(reportStr)
        return None
    
    def printStats(self, output=None, overwrite=False, precision=4, header=True):
        if isinstance(output,str):
            output=output
            if os.path.isfile(output):
                if overwrite:
                    file = open(output, 'w')
                    self.printData(output, precision, header)
                else:
                    file = open(output, 'a')
                    self.printData(file, precision, header)
            else:
                try:
                    os.makedirs(os.path.dirname(output))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
                if header:
                    file = open(output, 'w')
                else: 
                    file = open(output, 'a')
                self.printData(file, precision, header)
                
    def resultVector(self, order=None, precision='.3f'):
        #Order is a list of marker names that must match those contained in the sample
        #It is then used to get the reported ratios in that order
        if order is not None:
            if not isinstance(order, list):
                raise TypeError("Order must be a list (of str)")
                if not all([isinstance(i,str) for i in order]):
                    raise TypeError("The elements in order must be str")
        gatesInSample = [gate[0] for gate in self.vGates]
        if order is not None:
            #Assert equal amount of specified gates as there are gates in sample
            assert len(gatesInSample) == len(order)
            #Assert the gates in the sample are the same as those in order
            assert all([population in order for population in gatesInSample])
        else:
            #If no order specified, order is gates in the sample
            order=gatesInSample
        
        orderIndex=[]
        for gateName in order:
            index = gatesInSample.index(gateName)
            orderIndex.append(index)
        assert len(orderIndex) == len(self.vGates)
        result = np.empty(shape=len(self.vGates))
        outputCounter=0
        for i in orderIndex:
            nOfEvents = len(self.vGates[i][1]())
            nOfParentEvents = len(self.vGates[i][1].getParent())

            if nOfEvents == 0 or nOfParentEvents == 0:
                fraction=float(0)
            else:
                fraction=float(format(nOfEvents/nOfParentEvents,precision))

            result[outputCounter]=fraction
            outputCounter+=1
        return result
        
import tempfile
class AGQC:
    tmpFiles=[]
    
    def __init__(self, fcs, *args, **kwargs):
        pass
    
    def __call__(self, fcs, *args, **kwargs):
        if len(self.tmpFiles)==0:
            sys.stderr.write("Initiating QC files\n")
            for gate in fcs.vGates:
                tf = self.returnTempFile(fcs)
                self.printDownSample(tf,gate)
    
    #Files are opened in r+w mode as per default. Call seek(0) to access for read when complete
    #Creates a tempfile for the gate if it doesn't already exists
    #Then returns that filehandle
    def returnTempFile(self, gate):
        if not str(gate.name+gate.parentName) in self.tmpFiles[:,2]:
            tf = tempfile.NamedTemporaryFile(prefix="AG")
            name=tf.name
            print(name)
            gates=str(gate.name+gate.parentName)
            self.tmpFiles.append(tf, name, gates)
            return tf
        else:
            index = self.tmpFiles.index(str(gate.name+gate.parentName))
            return self.tmpFiles[index][0]
        
            
class AGExperiment:
    fcsList=[]              #LIST (str filepaths)
    normaliseLevel=None     #STR
    nOfPlates=0             #INT    
    normalise=None          #BOOL
    plateList=[]            #LIST (str filepaths)
    sampleList=[]           #LIST OF AGSample OBJECTS
    lFilter=None
    lMask=None
    lIgnoreTypes=None
    lMarkers=None
    comp_matrix=None
    man_comp=False
    compensation_exceptions=None
    flaggedSamples=[]       #List of tuple (path, flagged pop)
    resultHeader=None
    resultMatrix=None
    
    
    def __init__(self, experimentRoot, *args, **kwargs):
        if 'filters' in kwargs:
            if not isinstance(kwargs['filters'],str):
                if not isinstance(kwargs['filters'],list):
                    raise TypeError("if filters is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['filters']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in filters")
                    self.lFilter=kwargs['filters']
            else:
                self.lFilter=[kwargs['filters']]
            reportStr=str(len(self.lFilter))+" filter(s) defined\n"
            sys.stderr.write(reportStr)
            
        if 'mask' in kwargs:
            if not isinstance(kwargs['mask'],str):
                if not isinstance(kwargs['mask'],list):
                    raise TypeError("if mask is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['mask']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in mask")
                    self.lMask=kwargs['mask']
            else:
                self.lMask=[kwargs['mask']]
            reportStr=str(len(self.lMask))+" mask(s) defined\n"
            sys.stderr.write(reportStr)
            
        if 'ignoretype' in kwargs:
            if not isinstance(kwargs['ignoretype'],str):
                if not isinstance(kwargs['ignoretype'],list):
                    raise TypeError("if ignoretype is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['ignoretype']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in mask")
                    self.lIgnoreTypes=kwargs['ignoretype']
            else:
                self.lIgnoreTypes=[kwargs['ignoretype']]
            reportStr=str(len(self.lIgnoreTypes))+" ignoretype defined\n"
            sys.stderr.write(reportStr)
            
        if 'markers' in kwargs:
            if not isinstance(kwargs['markers'],str):
                if not isinstance(kwargs['markers'],list):
                    raise TypeError("if markers is passed, it must be a string or list of strings")
                else:
                    for item in kwargs['markers']:
                        if not isinstance(item,str):
                            raise TypeError("non-string element in markers")
                    self.lMarkers=kwargs['markers']
            else:
                self.lMarkers=[kwargs['markers']]

            
        if 'normaliseOn' in kwargs:
            if not isinstance(kwargs['normaliseOn'],str):
                raise (TypeError("normaliseOn must be string"))
            self.normalise=True
            self.normaliseLevel=kwargs['normaliseOn']
            reportStr="Plate normalisation/outlier detection requested\n"
            sys.stderr.write(reportStr)

        if 'compMatrix' in kwargs:
            if not isinstance(kwargs['compMatrix'],np.ndarray):
                raise TypeError("compensation matrix must be np array if passed")
            self.comp_matrix=kwargs['compMatrix']
            self.man_comp=True
            reportStr="Manual compensation specified\n"
            sys.stderr.write(reportStr)
        
        if 'compList' in kwargs:
            if not isinstance(kwargs['compList'],list):
                raise TypeError("compList must be passed as list of tuples (file, file with compensation)")
            else:
                for elem in kwargs['compList']:
                    if not isinstance(elem, tuple):
                        raise TypeError("elements in complist must be tuples (file, file with compensation)")
            if self.man_comp:
                raise ValueError("Cannot both give compensation exception list and provide manual compensation matrix")
            self.compensation_exceptions=kwargs['compList']

        #If no normalisation, collect all files
        self.fcsList=ag.collectFiles(experimentRoot,self.lFilter,self.lMask,self.lIgnoreTypes)
        if self.normalise:
            #Otherwise (plate normalisation) collect abs path to all end-of-tree folders
            self.plateList=ag.listDir(experimentRoot)
            self.checkPlateList()
            reportStr="Files are distributed in "+str(len(self.plateList))+" plates\n"
            sys.stderr.write(reportStr)
            
            
    def checkPlateList(self):
        #Function that goes through the list of folders and removes those that do not contain fcs files
        #The remaining folders are considered plates
        lFlaggedFolders=[]
        for index,folder in enumerate(self.plateList):
            filesInFolder=next(os.walk(folder))[2]
            if not filesInFolder:
                lFlaggedFolders.append(index)
        #To not get issues with cascading indicies while deleting go from the last element->first
        lFlaggedFolders=np.sort(lFlaggedFolders)[::-1]
        for index in lFlaggedFolders:
            del self.plateList[index]
    
    def apply(self, strategy, *args, **kwargs):
        if not hasattr(strategy, '__call__'):
            raise TypeError("Passed strategy does not seem to be a function")
                
        if self.normalise:
            reportStr="Applying strategy platewise\n"
            sys.stderr.write(reportStr)
            for plate in self.plateList:
                plate_samples=[]
                gc.collect()
                plate_samples=self.parse_folder(strategy, plate, *args)
                self.check_plate(plate_samples)
                for sample in plate_samples:
                    if self.resultHeader is None:
                        self.initResultMatrix(sample)
                    self.collectGateData(sample)
                #self.sampleList.extend(plate_samples)
        else:
            reportStr="Applying strategy\n"
            sys.stderr.write(reportStr)
            if True: #QC FLAG
                QCfile=open("AGQC_tmp.npy","w+b")
            for fcs in self.fcsList:
                if 'ag_non_verbose' in kwargs:
                    ag_non_verbose = kwargs['ag_non_verbose']
                else:
                    ag_non_verbose = False
                if self.compensation_exceptions is not None:
                    comp_matrix=self.check_compensation_exception(fcs)
                if self.man_comp:
                    comp_matrix=self.comp_matrix
                sample = ag.loadFCS(fcs, return_type="AGSample", comp_matrix=comp_matrix, non_verbose=ag_non_verbose, markers=["IgA", "CD27" ,"CD34" ,"CD19", "IgD" ,"CD45","CD38","CD24"])
                if sample is None:
                    continue
                gatedSample=self.fcs_apply_strategy(sample,strategy, *args, **kwargs)
                if self.resultHeader is None:
                        self.initResultMatrix(sample)
                self.collectGateData(gatedSample)
                #gatedSample.fcsDF=[] #Clear DF to save memory
                self.sampleList.append(gatedSample.sample)
                if True:#QC Flag
                    self.printDownsample(QCfile,gatedSample)
            if True: #QC Flag
                QCfile.close()
        nOfFlagged=len(self.flaggedSamples)
        fhandle=open("testRun.log.txt",'w')
        reportStr="Complete, "+str(nOfFlagged)+" samples flagged for inspection\n"
        sys.stderr.write(reportStr)
        fhandle.write(reportStr)
        for elem in self.flaggedSamples:
            reportStr=elem[0]+"\n"+elem[1]+"\n"
            sys.stderr.write(reportStr)
            fhandle.write(reportStr)
    
    def printDownsample(self, file, sample):
        flattenedArr=sample.downSample.flatten()
        arr=np.array([sample.sample, '16'])
        result=np.append(arr,flattenedArr)
        np.save(file,result)        
        return
    
    def initResultMatrix(self, fcs):
        assert len(fcs.vGates) > 0
        self.resultMatrix=[]
        self.resultHeader=["sampleName"]
        for gate in fcs.vGates:
            self.resultHeader.extend([gate.name, gate.name+str("/")+gate.parentName])
            
    def collectGateData(self, fcs):
        assert all(isinstance(i, list) for i in [self.resultHeader, self.resultMatrix])
        #Create empty python list of correct size
        sampleResults=[None]*len(self.resultHeader)
        sampleResults[0] = fcs.sample
        for gate in fcs.vGates:
            indexOfGate=self.resultHeader.index(gate.name)
            if indexOfGate < 0:
                reportStr="sample contain unknown gates, skipping\n"
                sys.stderr.write(reportStr)
                sys.stderr.write(fcs.filePath)
                return
            currentGate=float(len(gate.current))
            currentGateParent=len(gate.parent)
            if currentGateParent==0:
                currentRatio="NA"
            else:
                currentRatio=float(len(gate.current)/len(gate.parent))
            sampleResults[indexOfGate] = currentGate
            sampleResults[indexOfGate+1] = currentRatio
        self.resultMatrix.append(sampleResults)
    
    def printExperiment(self, file):
        assert isinstance(file, str)
        assert all(isinstance(i, list) for i in [self.resultHeader, self.resultMatrix])
        fhandle = open(file, 'w')
        for elem in self.resultHeader:
            outputStr=str(elem)+"\t"
            fhandle.write(outputStr)
        fhandle.write("\n")
        for sample in self.resultMatrix:
            for elem in sample:
                outputStr=str(elem)+"\t"
                fhandle.write(outputStr)
            fhandle.write("\n")
        fhandle.close()
        
    def parse_folder(self, strategy, folder, comp_matrix=None, *args, **kwargs):
        if 'ag_non_verbose' in kwargs:
            ag_non_verbose = kwargs['ag_non_verbose']
        else:
            ag_non_verbose = False
        reportStr="Parsing plate: "+folder+" \n"
        sys.stderr.write(reportStr)
        fcs_in_folder = ag.collectFiles(folder, lFilter=self.lFilter, lMask=self.lMask, lIgnoreTypes=self.lIgnoreTypes)
        samples_in_plate=[]
        for fcs in fcs_in_folder:
            if self.compensation_exceptions is not None:
                    comp_matrix=self.check_compensation_exception(fcs)
            if self.man_comp:
                comp_matrix=self.comp_matrix
            sample = ag.loadFCS(fcs, return_type="AGSample", comp_matrix=comp_matrix, non_verbose=ag_non_verbose, markers=["IgA", "CD27" ,"CD34" ,"CD19", "IgD" ,"CD45","CD38","CD24"])
            if sample is None:
                continue
            sample = self.fcs_apply_strategy(sample, strategy, *args, **kwargs)
            if self.resultHeader is None:
                self.initResultMatrix(sample)
            sample.fcsDF=[]
            samples_in_plate.append(sample)
        return samples_in_plate
    
    def fcs_apply_strategy(self, fcs, strategy, *args, **kwargs):
        fcs = strategy(fcs, *args, **kwargs)
        return fcs
    
    def check_plate(self, plate_samples, order=None):
        #print(plate_samples)
        #population_names=[]
        #population_avgs=[]
        #Gets list of AGSamples in this plate.
        #Check for outliers, compute averages and report
        nOfGates=[len(sample.vGates) for sample in plate_samples]
        if not nOfGates[1:] == nOfGates[:-1]:
            sys.stderr.write("plate samples has unequal amount of gates\n")
            raise  
        if len(nOfGates)==0:
            reportStr="Sample object has no gates\n"
            sys.stderr.write(reportStr)
            return None
        
        nOfGates=nOfGates[0]
        popList=[]
        for sample in plate_samples:
            popList.append(sample.resultVector(order=order))
        
        plate_matrix=np.array(popList, dtype=np.float64)
        #print(plate_matrix)
        #print("shape: "+str(plate_matrix.shape[1]))
        #iterate through cols, compute means and std dev
        t_plate_matrix=np.transpose(plate_matrix)
        for i in np.arange(0,len(t_plate_matrix),1):
            #Important, require array to be C continous with order='C'
            np_popList=np.asarray(t_plate_matrix[i], dtype=np.float64, order='C')
            #print(np_popList)
            mean, var = ag.Stat_GetMeanAndVariance_double(np_popList)
            #print("mean: "+str(mean)+"\nstd_dev: "+str(np.sqrt(var)))
            std_dev = np.sqrt(var)
            for x in np.arange(0, len(np_popList),1):
                if np_popList[x] < (mean - 4*std_dev) or np_popList[x] > (mean + 2*std_dev):
                    self.flaggedSamples.append((plate_samples[x].filePath, plate_samples[x].vGates[i][0]))
        #np_popList=np.asarray(popList,dtype=np.float64)
        #mean, var = ag.Stat_GetMeanAndVariance_double(np_popList)
        return None
    
    def check_metadata_internal(self, filePath, fcsDF, lFlagged, metaDict):
        bOk=True
        lFlags=[]
        spill_matrix=metaDict['SPILL'].split(',')
        n = int(spill_matrix[0]) #number of colors
        colNames=fcsDF.columns[4:(n+4)] 
        
        #Check that the fcs has a supported fcs version
        try:
            fcs_vs = metaDict['__header__']['FCS format'].decode('UTF-8')
        except UnicodeError:
            reportStr="Unexpected encoding in metadata (not UTF-8), proceed with caution"
            bOk=False
            lFlags.append(reportStr)
        if not fcs_vs in ['FCS3.0','FCS2.0','FCS3.1']:
            reportStr="FCS file version ("+fcs_vs+") not guaranteed to work, procced with caution"
            bOk=False
            lFlags.append(reportStr)
            
        #Check that the first four fields in the fcs are forward and side scatters
        #Otherwise parsing will fail
        scatter_cols=fcsDF.columns[0:4]
        if not all([scatter in ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-H'] for scatter in scatter_cols]):
            reportStr="First four columns in fcs are not forward and side scatters"
            bOk=False
            lFlags.append(reportStr)
            
        #If markers have been passed to the experiment object, check that they are
        #correctly specified for each sample
        if self.lMarkers is not None:
            if not set(colNames) == set(self.lMarkers):
                reportStr="Specified markers not in fcs"
                bOk=False
                lFlags.append(reportStr)
                
        #Check that compensation data is available (non-zero)
        comp_matrix = np.array(spill_matrix[n+1:]).reshape(n, n).astype(float)
        tmp_identity = np.identity(n)
        if np.array_equal(comp_matrix, tmp_identity):
            reportStr="No compensation data available"
            bOk=False
            lFlags.append(reportStr)
            
        if not bOk:
            lFlags.append(filePath)
            lFlagged.append(lFlags[::-1])  
        return bOk
    
    def check_metadata(self, *args, **kwargs):
        if self.fcsList.empty:
            raise ValueError("This AGExperiment does not have any fcs files associated with it")
        bOk=True
        metadata_warnings=0
        reportStr="Checking metadata for all "+str(len(self.fcsList))+" fcs files in experiment:\n"
        sys.stderr.write(reportStr)
        lFlagged=[]
        progress_counter=0
        for fcs in self.fcsList:
            progress_counter+=1
            metadata,fcsDF = ag.parse(fcs,output_format='DataFrame')
            if metadata is None:
                metadata_warnings+=1
                lFlagged.append([fcs,"Could not parse metadata"])
                continue
            bOk=self.check_metadata_internal(fcs,fcsDF,lFlagged,metadata)
            if not bOk:
                metadata_warnings+=1
            if progress_counter%10==0:
                progress=progress_counter/len(self.fcsList)
                update_progress(progress)
        update_progress(1.0)
        reportStr=str(metadata_warnings)+" files had metadata flags\n"
        sys.stderr.write(reportStr)   
        if len(lFlagged) > 0:
            for sample in lFlagged:
                reportStr=sample[0]+"\nFlags:\n"
                sys.stderr.write(reportStr)
                for i in np.arange(1,len(sample),1):
                    reportStr=sample[i]+"\n"
                    sys.stderr.write(reportStr)            
    
    def check_compensation_exception(self, fcs):
        for i in np.arange(0, len(self.compensation_exceptions),1):
            if fcs == self.compensation_exceptions[i][0]:
                reportStr="Sample is in compensation exception list, collecting external compensation matrix\n"
                sys.stderr.write(reportStr)
                metadata, fcsDF = ag.loadFCS(self.compensation_exceptions[i][1], return_type="index", metadata=True, non_verbose=True)
                comp_matrix=ag.getCompensationMatrix(fcsDF, metadata)[1]
                if comp_matrix is not None:
                    reportStr="Succesfully collected external compensation matrix\n"
                    sys.stderr.write(reportStr)
                    return comp_matrix
                else:
                    reportStr=self.compensation_exceptions[i][1]+" has a zero or none compensation matrix\n"
                    raise ValueError("External compensation matrix in compList is zero")
        return None
    
def is_decade(x, base=10):
    if not np.isfinite(x):
        return False
    if x == 0.0:
        return True
    lx = np.log(np.abs(x)) / np.log(base)
    return is_close_to_int(lx)

def is_close_to_int(x):
    if not np.isfinite(x):
        return False
    return abs(x - nearest_int(x)) < 1e-10

def nearest_int(x):
    if x == 0:
        return int(0)
    elif x > 0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)

        
def convertToLogishPlotCoordinates(Ticlocs, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = ag.logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    tTiclocs=ag.logishTransform(Ticlocs, T)
    plotTics=[]
    for tTic in tTiclocs:
        plotTic=(tTic-tMinMax[0])/transformedRange*actualRange+vmin
        plotTics.append(plotTic)
    assert len(tTiclocs)==len(Ticlocs)
    return plotTics

def convertToLogishPlotCoordinate(Ticloc, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = ag.logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    tTicloc=ag.logishTransform([Ticloc], T)[0]
    plotTic=(tTicloc-tMinMax[0])/transformedRange*actualRange+vmin
    return plotTic

def invertLogishPlotcoordinates(plotTics, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = ag.logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    invPlotTics=[]
    for tTic in plotTics:
        invPlotTic=(tTic-vmin)/actualRange*transformedRange+tMinMax[0]
        invPlotTics.append(invPlotTic)
    result=ag.inverseLogishTransform(invPlotTics, T)
    return result

def invertLogishPlotcoordinate(plotTic, vmin, vmax, T):
    actualRange=vmax-vmin
    tMinMax = ag.logishTransform([vmin, vmax], T)
    transformedRange = tMinMax[1]-tMinMax[0]
    invPlotTic=(plotTic-vmin)/actualRange*transformedRange+tMinMax[0]
    result=ag.inverseLogishTransform([invPlotTic], T)[0]
    return result

class LogishLocator(Locator):
    """
    Determine the tick locations for logish axes based on LogLocator
    Hacked version of LogLogator that covers normal usecases of the logish scale
    Only defined with ticlocations for data in range -50000 < x < 1 000 000
    """

    def __init__(self, linCutOff=1000, subs=(1.0,), numdecs=4, numticks=None):
        """
        Place ticks on the locations : subs[j] * base**i
        Parameters
        ----------
        subs : None, string, or sequence of float, optional, default (1.0,)
            Gives the multiples of integer powers of the base at which
            to place ticks.  The default places ticks only at
            integer powers of the base.
            The permitted string values are ``'auto'`` and ``'all'``,
            both of which use an algorithm based on the axis view
            limits to determine whether and how to put ticks between
            integer powers of the base.  With ``'auto'``, ticks are
            placed only between integer powers; with ``'all'``, the
            integer powers are included.  A value of None is
            equivalent to ``'auto'``.
        """
        if numticks is None:
            if rcParams['_internal.classic_mode']:
                numticks = 15
            else:
                numticks = 'auto'
                
        self._base=np.exp(1)
        self.subs(subs)
        self.numdecs = numdecs
        self.numticks = numticks
        self.T = linCutOff
        
    def set_params(self, subs=None, numdecs=4, numticks=None):
        """Set parameters within this locator."""
        if subs is not None:
            self.subs(subs)
        if numdecs is not None:
            self.numdecs = numdecs
        if numticks is not None:
            self.numticks = numticks

    # FIXME: these base and subs functions are contrary to our
    # usual and desired API.

    def subs(self, subs):
        """
        set the minor ticks for the log scaling every base**i*subs[j]
        """
        if subs is None:  # consistency with previous bad API
            self._subs = 'auto'
        elif isinstance(subs, six.string_types):
            if subs not in ('all', 'auto'):
                raise ValueError("A subs string must be 'all' or 'auto'; "
                                 "found '%s'." % subs)
            self._subs = subs
        else:
            self._subs = np.asarray(subs, dtype=float)

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.view_limits()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if self.numticks == 'auto':
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks
        
    #if vmin < self.T:
        tmpTicLoc=[-50000, -40000, -30000, -20000, -10000, -5000,-4000,-3000, -2000, -1000, 0]
        Ticlocs = list(set(np.clip(tmpTicLoc, vmin, self.T)))
        Ticlocs = list(np.sort(Ticlocs))
    
    #if vmax > self.T:
        tmpTicLoc=[0,1000]
        tmpTicLoc.extend(np.arange(1000.0,10001,1000))
        tmpTicLoc.extend(np.arange(10000.0,100001,10000))
        tmpTicLoc.extend(np.arange(100000.0,1000000,100000)) #[10000.0,100000.0, 200000, 300000, 1000000.0]
        Ticlocs.extend(tmpTicLoc)
        clip_Ticlocs=list(set(np.clip(Ticlocs,vmin, vmax)))
        Ticlocs=np.sort(clip_Ticlocs)
        #ADD HOC POSSIBLY
        Ticlocs=Ticlocs[1:-1]
        Ticlocs=convertToLogishPlotCoordinates(Ticlocs, vmin, vmax, self.T)
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        return self.raise_if_exceeds(np.asarray(Ticlocs))

    def view_limits(self, vmin=None, vmax=None):
        'Try to choose the view limits intelligently'
        vmin, vmax = self.axis.get_view_interval()
        return vmin, vmax


class LogishFormatter(Formatter):
    """
    Base class for formatting ticks on a logish scale.
    Hacked version of LogFormatter that covers normal usecases of the logish scale
    Only defined with formatting ticlabels for data in range -50000 < x < 1 000 000

    
    Parameters
    ----------
    labelOnlyBase : bool, optional, default: False
        If True, label ticks only at integer powers of base.
        This is normally True for major ticks and False for
        minor ticks.
    minor_thresholds : (subset, all), optional, default: (1, 0.4)
        If labelOnlyBase is False, these two numbers control
        the labeling of ticks that are not at integer powers of
        base; normally these are the minor ticks. The controlling
        parameter is the log of the axis data range.  In the typical
        case where base is 10 it is the number of decades spanned
        by the axis, so we can call it 'numdec'. If ``numdec <= all``,
        all minor ticks will be labeled.  If ``all < numdec <= subset``,
        then only a subset of minor ticks will be labeled, so as to
        avoid crowding. If ``numdec > subset`` then no minor ticks will
        be labeled.
    linthresh : float, optional, default: 1000
        The threshold for the logicle scale change from linear-like to log-like scaling
 
    Notes
    -----
    The `set_locs` method must be called to enable the subsetting
    logic controlled by the ``minor_thresholds`` parameter.
    In some cases such as the colorbar, there is no distinction between
    major and minor ticks; the tick locations might be set manually,
    or by a locator that puts ticks at integer powers of base and
    at intermediate locations.  For this situation, disable the
    minor_thresholds logic by using ``minor_thresholds=(np.inf, np.inf)``,
    so that all ticks will be labeled.
    To disable labeling of minor ticks when 'labelOnlyBase' is False,
    use ``minor_thresholds=(0, 0)``.  This is the default for the
    "classic" style.
    Examples
    --------
    To label a subset of minor ticks when the view limits span up
    to 2 decades, and all of the ticks when zoomed in to 0.5 decades
    or less, use ``minor_thresholds=(2, 0.5)``.
    To label all minor ticks when the view limits span up to 1.5
    decades, use ``minor_thresholds=(1.5, 1.5)``.
    """
    def __init__(self, labelOnlyBase=False,
                 minor_thresholds=None,
                 linthresh=1000):
        
        self.labelOnlyBase = labelOnlyBase
        if minor_thresholds is None:
            if rcParams['_internal.classic_mode']:
                minor_thresholds = (0, 0)
            else:
                minor_thresholds = (1, 0.4)
        self.minor_thresholds = minor_thresholds
        self._sublabels = None
        self._linthresh = linthresh
        self._base = np.exp(1)



    def _num_to_string(self, x, vmin, vmax):
        x = round(x,0)
        if not x in [-5000, -4000, -3000, -2000, -1000, 0 ,1000,10000,100000,1000000]:
            s = ''
        else:
            s = self.pprint_val(x, vmax - vmin)
        return s

    def __call__(self, x, pos=None):
        """
        Return the format for tick val `x`.
        """
        if x == 0.0:  # Symlog
            return '0'
        vmin, vmax = self.axis.get_view_interval()
        #tVals = ag.logishTransform([vmin, vmax, x], self._linthresh)
        # only label the decades
        #fx = (x-vmin)/(vmax-vmin)*(tVals[1] - tVals[0])-tVals[0]
        #fx = ag.inverseLogishTransform([fx],self._linthresh)[0]
        fx=invertLogishPlotcoordinate(x,vmin,vmax,self._linthresh)
        #print(fx)
        s = self._num_to_string(fx, vmin, vmax)
        return self.fix_minus(s)
        

    def pprint_val(self, x, d):
        #If the number is at or below the set lin-cutoff (_lintrehsh)
        #Print it as an int
        #TODO: WHY DO I NEED THE +1 HERE?
        if x <= self._linthresh+1:
            return '%d' % x

        fmt = '%1.3e'
        s = fmt % x
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            if exponent:
                if float(mantissa) > 1:
                    s = '$%s*10^{%d}$' % (mantissa, exponent)
                else:
                    s = '$%s0^{%d}$' % (mantissa, exponent)
            else:
                s = mantissa
        else:
            s = s.rstrip('0').rstrip('.')
        return s