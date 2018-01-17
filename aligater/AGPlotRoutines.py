#	______|\__________\o/__________
#			~~aliGater~~
#	(semi)automated gating software
#
#	Utilizing Dislin for plots, non-commercial use only!
#	Relevant EULA section:
#	Grant of Free Usage
#	You are allowed to use DISLIN for free as a private person or as a member of an institute that does not earn money with selling any products and services.
#	http://dislin.de
#
#	Utilizing Intels Math Kernel Library (MKL)
#
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	http://nilssonlab.org
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import aligater as ag
import sys

sentinel = object()
def plotHeatmap(fcsDF, x, y, vI=sentinel, bins=300):
    if vI is sentinel:
        vI=fcsDF.index
    elif len(vI)==0:
        sys.stderr.write("Passed index contains no events")
        return None
    
    vX=ag.getGatedVector(fcsDF, x, vI)
    vY=ag.getGatedVector(fcsDF, y, vI)
    
    matplotlib.rcParams['image.cmap'] = 'jet'
    heatmap, xedges, yedges = np.histogram2d(vX, vY, bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap=np.ma.masked_where(heatmap == 0, heatmap)
    
    plt.clf()
    fig, ax = plt.subplots()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.xlabel(x)
    plt.ylabel(y)
    cmap=plt.get_cmap()
    cmap.set_bad(color='white') #Zeroes should be white, not blue
    plt.show()
    return fig, ax

def addLine(fig, ax, lStartCoordinate, lEndCoordinate, size=2):
    ax.plot([lStartCoordinate[0], lEndCoordinate[0]], [lStartCoordinate[1], lEndCoordinate[1]], color='r', linestyle='-', linewidth=size)
    return fig

def addArrow(fig, ax, lStartCoordinate, lEndCoordinate, size=5000):
    ax.arrow(lStartCoordinate[0], lStartCoordinate[1], lEndCoordinate[0]-lStartCoordinate[0], lEndCoordinate[1]-lStartCoordinate[1], head_width=size, head_length=size, fc='r', ec='r')
    return fig

def main():
	return None

if __name__ == '__main__':
	main()