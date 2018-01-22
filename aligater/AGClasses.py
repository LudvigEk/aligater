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
import math
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms

class LogishScale(mscale.ScaleBase):
    name = 'logish'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)
        self.thresh = None #thresh

    def get_transform(self):
        return self.CustomTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):
        pass

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            return <Transform the stuff here>

        def inverted(self):
            return LogishScale.InvertedCustomTransform(self.thresh)

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            return <reverse transform here>

        def inverted(self):
            return LogishScale.CustomTransform(self.thresh)

# Register new scale
mscale.register_scale(LogishScale)