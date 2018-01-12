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

sentinel = object()
def plotHeatmap(fcsDF, x, y, vI=sentinel, bins=300):
	if vI is sentinel:
		vI=fcsDF.index
	matplotlib.rcParams['image.cmap'] = 'jet'
	vX=ag.getGatedVector(fcsDF, x, vI)
	vY=ag.getGatedVector(fcsDF, y, vI)
	heatmap, xedges, yedges = np.histogram2d(vX, vY, bins=300)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	heatmap=np.ma.masked_where(heatmap == 0, heatmap)
	plt.clf()
	plt.imshow(heatmap.T, extent=extent, origin='lower')
	cmap=plt.get_cmap()
	cmap.set_bad(color='white') #Zeroes should be white, not blue
	plt.show()

if __name__ == '__main__':
	main()

def main():
	return None
