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
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import matplotlib.ticker as ticker

class CustomScale(mscale.ScaleBase):
    name = 'logish'

    def __init__(self, axis, linCutOff=1000, **kwargs):
        self.linCutOff = linCutOff #default treshold
        mscale.ScaleBase.__init__(self)
        self.thresh=None
        
    def get_transform(self):
        return self.CustomTransform(self.thresh, self.linCutOff)

    
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(HlogMajorLocator())
        axis.set_major_formatter(LogFormatterMathtext(10))
        axis.set_minor_locator(HlogMinorLocator())
        axis.set_minor_formatter(NullFormatter()) 
        pass

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        has_inverse = True
        is_separable = True

        def __init__(self, thresh, linCutOff):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.linCutOff = linCutOff

        def transform_non_affine(self, a):
            tA = np.empty_like(a)
            a_idx=0
            while a_idx < len(a):
                if a[a_idx] >= self.linCutOff:
                   tA[a_idx] = np.log(10 * a[a_idx] / self.linCutOff - 10)
                else:
                    tA[a_idx] = (a[a_idx]/self.linCutOff + np.log(10.0) - 1)/np.log(10)
                a_idx+=1
            return tA

        def inverted(self):
            return CustomScale.InvertedCustomTransform(self.thresh,self.linCutOff)

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, thresh, linCutOff):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.linCutOff = linCutOff
            
        def transform_non_affine(self, a):
            invA=np.empty_like(a)
            a_idx=0
            while a_idx < len(a):
                if a[a_idx] >= np.log(10):
                    invA[a_idx]= (np.exp(a[a_idx])+10)*self.linCutOff/10
                else:
                    invA[a_idx] = self.linCutOff*(np.log(10.0)*a[a_idx] - np.log(10.0) + 1)
                a_idx+=1
            return invA

        def inverted(self):
            return CustomScale.CustomTransform(self.thresh,self.linCutOff)

# Register new scale
mscale.register_scale(CustomScale)