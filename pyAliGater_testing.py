#!/usr/bin/env python3
import aligater as ag
    
def gateBCellDataSet(fcs):
    no_clutter=ag.gateThreshold(fcs(),"FSC-A", "FSC-H",thresh=260000, orientation='vertical',population="lower", plot=False)
    singlets=ag.gatePC(fcs(),"FSC-A", "FSC-H",center='density',widthScale=4, heightScale=4, vI=no_clutter, plot=False)
    fcs.update(ag.AGgate(singlets, fcs.full_index()),"singlets")
    
    PBMCstep1=ag.gateThreshold(fcs(),xCol="FSC-A",yCol="SSC-A", vI=singlets,orientation="horisontal",thresh=70000,population="lower", plot=False)
    PBMC=ag.gatePC(fcs(),"FSC-A","SSC-A",center='centroid',widthScale=3, heightScale=2, vI=PBMCstep1, plot=False)
    fcs.update(ag.AGgate(PBMC, singlets),"PBMC")
    
    mean, sigma, maxVal=ag.axisStats(fcs(), "CD34", PBMC)
    CD45step1 = ag.gateThreshold(fcs(), "CD34", "CD45", orientation='vertical', vI=PBMC,population="lower",thresh=3*sigma, scale='logish',plot=False)
    lim = ag.valleySeek(fcs(), "CD45",vI=CD45step1,interval=[0,2500],sigma=2,bins=1000)
    CD45pos = ag.gateThreshold(fcs(), "CD34", "CD45", vI=CD45step1, orientation='horisontal',thresh=lim,scale='logish',plot=False) 
    fcs.update(ag.AGgate(CD45pos, PBMC),"CD45")
    
    CD34step1=ag.gateThreshold(fcs(), "CD34", "CD45", orientation='vertical', vI=PBMC,population="upper",thresh=3*sigma, scale='logish',plot=False)
    lim=ag.valleySeek(fcs(), "CD45",vI=CD34step1,interval=[500,2500],sigma=2,bins=200)
    CD34pos=ag.gateThreshold(fcs(), "CD34", "CD45", orientation='horisontal', vI=CD34step1,population="lower",thresh=lim, scale='logish',plot=False)
    fcs.update(ag.AGgate(CD34pos, PBMC),"CD34")

    lim=ag.valleySeek(fcs(),xCol="CD19",vI=CD45pos,interval=[0,2000],bins=1000,sigma=2, scale='logish')
    CD19pos=ag.gateThreshold(fcs(),"CD19","CD45",vI=CD45pos,thresh=lim,scale='logish',orientation='vertical',population='upper',plot=False)
    fcs.update(ag.AGgate(CD19pos, CD45pos),"CD19")

    xlim=ag.valleySeek(fcs(),"IgD", CD19pos, interval=[500,2500],bins=300,sigma=2, scale='logish')
    ylim=ag.valleySeek(fcs(),"CD27", CD19pos, interval=[400,700],bins=300,sigma=2, scale='logish')
    switchB, preSwitchB, naiveB, exhaustedB,solution = ag.variableQuadGate(fcs(), "IgD", "CD27", [xlim, xlim, ylim, ylim], testRange=[4000,6000], testSteps=10, position='left', vI=CD19pos,scale='logish', plot=False, only_solution=False)
    fcs.update(ag.AGgate(switchB, CD19pos),"switchB")
    fcs.update(ag.AGgate(preSwitchB, CD19pos),"preSwitchB")
    fcs.update(ag.AGgate(naiveB, CD19pos),"naiveB")
    fcs.update(ag.AGgate(exhaustedB, CD19pos),"exhaustedB")
    
    mean, sigma, maxVal = ag.axisStats(fcs(), "CD38", vI=switchB)  
    plasmablasts=ag.gateCorner(fcs(), "CD24", "CD38",1000,mean+sigma,"lower","upper", vI=switchB,scale='logish',plot=False)
    fcs.update(ag.AGgate(plasmablasts, switchB),"plasmablasts")
    
    xmean,xsigma,xmaxVal = ag.axisStats(fcs(),xCol="CD24",vI=naiveB)
    ymean,ysigma,ymaxVal = ag.axisStats(fcs(),xCol="CD38",vI=naiveB)
    transitionals=ag.gateCorner(fcs(),"CD24", "CD38",xThresh = xmaxVal, yThresh=ymaxVal+2000, vI=naiveB, scale='logish',plot=False)
    fcs.update(ag.AGgate(transitionals, naiveB),"transitionals")
    
    lim = ag.valleySeek(fcs(),"IgA",vI=switchB,interval=[750,2000],bins=300, sigma=1, scale='logish')
    IgApos=ag.gateThreshold(fcs(), "IgA","CD19", orientation='vertical', vI=switchB, thresh=lim, scale='logish',plot=False)
    fcs.update(ag.AGgate(IgApos, switchB),"IgA")
    return fcs

fcsList=ag.collectFiles("/media/ludvig/Project_Storage/GAIM/data/GAIM_fcs/03-03-2016", lFilter=["B cells"])
count=0
for fcs in fcsList:
    fcs=ag.loadFCS(fcs, compensate=True, return_type="AGsample")
    fcs = gateBCellDataSet(fcs)
    if count==0:
        fcs.printStats("nånstans",header=True)
    else:
        fcs.printStats("nånstans",header=False)
    count+=1
    