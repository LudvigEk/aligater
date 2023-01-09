#!/usr/bin/env Python3

import aligater as ag

#This function simply creates a list of [x,y] coordinates from a given [xstart,ystart] to
#[xend, yend] with even spacing between
#Such lists can be supplied to gatePointList
def create_vPL(startx,endx, starty,endy, bins=300):
    x_coord=[]
    y_coord=[]
    xstep=(endx-startx)/bins
    ystep=(endy-starty)/bins
    for i in ag.np.arange(startx,endx,xstep):
        x_coord.append(i)
    for i in ag.np.arange(starty,endy,ystep):
        y_coord.append(i)
    assert len(x_coord)==len(y_coord)
    vPL = ag.np.column_stack([x_coord,y_coord])
    return vPL

def example_gating_strategy(fcs, *args):

	#Defining some comfy marker aliases
	#This allows to use these shorthand variables instead of the whole string when calling functions
	SSC="SSC 488/10-A"
	FSC="FSC 488/10-A"
	CD39="BB515 CD39-A"
	CD25="PE-Cy7 CD25-A"
	CD127="PE CD127-A"
	CCR6="PE-Dazzle 594 CCR6-A"
	HLADR="BV650 HLA-DR-A"
	CCR7="BV711 CCR7-A"
	CXCR5="BV786 CXCR5-A"
	CXCR3="BV421 CXCR3-A"
	CD194="BV605 CD194-A"
	CD4="BV510 CD4-A"
	CD3="Alexa Fluor 700 CD3-A"
	CD8="APC-H7 CD8-A"
	CD45RA="APC CD45RA-A"


	#Treshold gate to remove high FSC-H events
	no_clutter1=ag.gateThreshold(fcs,"no_clutter","FSC 488/10-A", "FSC 488/10-H",thresh=214000, orientation='vertical',population="lower")
	no_clutter=ag.gateThreshold(fcs,"no_clutter","FSC 488/10-A", "FSC 488/10-H", parentGate=no_clutter1,thresh=214000, orientation='horizontal',population="lower")

	#Horizontal path calculates a Dijstra's shortest path with constraints from startY to endY
	#The constraints in this case are:
	#1) Initial 'zoom in' on SSC-A between 20,000 and 140,000
	#2) maxstep 2
	#3) phi 0.1
	#4) direction 'both'
	#I.e. The path can move both up and down as it traverses, for each step on the x axis it can take a maximum of +-2 steps in the y direction
	#Longer steps are slightly penalized over shorter steps
	PBMCstep1 = ag.horizontalPath(fcs, name="PBMCstep1", xCol=FSC, yCol=SSC, parentGate=no_clutter, population='lower',
	             startY=40000,endY=60000, xboundaries=None, yboundaries=[20000,140000], leftRight=True , direction='both', maxStep=2, phi=0.1,
	             bins=100, sigma=1, scale='linear', T=1000)

	#To remove debris in the bottom-left corner, a custom fixed gate is defined through gatePointList
	#First a list, "vPL", of x and y coordinates is defined then supplied to the function
	vPL=[[55000,0],[55000,35000]]
	vPL2 = create_vPL(55000,70000,35000,50000)
	vPL3 = create_vPL(70000,90000,50000,120000)
	vPL.extend(vPL2)
	vPL.extend(vPL3)
	vI=ag.gatePointList(fcs(), xCol=FSC, yCol=SSC, vPL=vPL, population='upper',vI=PBMCstep1(), bhorizontal=False)
	#gatePointList is a lower lever function and that doesn't return aligater.AGSample objects
	#It instead returns a list-like index, which points to events in the sample dataframe that pass the gate
	#This is useful to known if custom gates are constructed
	#Convert an list-like integer index to the pandas dataframe of the AGSample to an AGGate like this:
	PBMCstep2=ag.AGgate(vI, PBMCstep1, xCol=FSC, yCol=SSC, name="PBMCstep2")

	#Finally to further defined the PBMC cluster a principal component based ellipsoid gate is drawn
	#This behaves similarly to a 2-D gaussian mixed model with some useful constraints
	#In this case the semiaxis of the ellipsoid is scaled to 4 and 3.5 times the length of the first two eigenvectors through the widthScale and heightScale arguments
	#The angle of the ellipsoid is also slightly nudged up by 3 degrees through the adjustAngle Argument
	PBMC=ag.gatePC(fcs,name="PBMC",xCol=FSC,yCol=SSC,center='centroid', adjustAngle=3,widthScale=4, heightScale=3.5, parentGate=PBMCstep2)

	fcs.update(PBMC, QC=False, MFI=False)

	singlets=ag.gatePC(fcs,xCol=FSC, yCol="FSC 488/10-H",name="singlets",center='density', adjustAngle=1, widthScale=6, heightScale=4.5, parentGate=PBMC)
	fcs.update(singlets, QC=False, MFI=False)


	#CD3pos/neg
	xlim=ag.valleySeek(fcs=fcs, xCol=CD3, parentGate=singlets, interval=[0,1000], require_local_min=True, scale='bilog',T=200)
	if xlim == ag.np.inf:
	    xlim=400
	CD3pos = ag.gateThreshold(fcs=fcs, name='CD3pos', xCol=CD3, yCol="FSC 488/10-H", parentGate=singlets, thresh=xlim, population='upper', scale='bilog', T=200)
	CD3neg = ag.gateThreshold(fcs=fcs, name='CD3pos', xCol=CD3, yCol="FSC 488/10-H", parentGate=singlets, thresh=xlim, population='lower', scale='bilog', T=200)
	fcs.update(CD3pos, QC=False, MFI=False)
	fcs.update(CD3neg, QC=False, MFI=False)

	#CD4/CD8
	xlim=ag.valleySeek(fcs, xCol=CD8, parentGate=CD3pos, interval=[0,3000], require_local_min=True, scale='bilog',T=200)
	if xlim == ag.np.inf:
	    xlim=1500
	CD8neg_tmp = ag.gateThreshold(fcs=fcs, name='CD8neg_temp', xCol=CD8, parentGate=CD3pos, thresh=xlim, population='lower', scale='bilog',T=200)
	ylim=ag.valleySeek(fcs, xCol=CD4, parentGate=CD8neg_tmp, interval=[50,1000], require_local_min=True, scale='bilog',T=200)
	if ylim == ag.np.inf:
	    ylim=200
	if ylim < 100: # shouldnt flop down all the way here
	    ylim=200 #set to ~fmo
	CD4pos,CD4CD8_doublepos,CD8pos,CD4CD8_doubleneg = ag.quadGate(fcs, names=["","", "", ""], xCol=CD8, yCol=CD4, parentGate=CD3pos, xThresh=xlim, yThresh=ylim, scale='bilog',T=200)
	fcs.update(CD4pos, QC=False, MFI=False)
	fcs.update(CD8pos, QC=False, MFI=False)
	fcs.update(CD4CD8_doublepos, QC=False, MFI=False)
	fcs.update(CD4CD8_doubleneg, QC=False, MFI=False)

	return fcs