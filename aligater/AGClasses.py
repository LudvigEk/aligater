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
from matplotlib.ticker import Locator, Formatter
import math
import aligater as ag
from matplotlib import rcParams, cbook
import six
import sys

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

def decade_down(x, base=10):
    'floor x to the nearest lower decade'
    if x == 0.0:
        return -base
    lx = np.floor(np.log(x) / np.log(base))
    return base ** lx

def decade_up(x, base=10):
    'ceil x to the nearest higher decade'
    if x == 0.0:
        return base
    lx = np.ceil(np.log(x) / np.log(base))
    return base ** lx

def nearest1k(x):
    if x < 0:
        return np.floor(x/1000)*1000
    if x >= 0:
        return 0
    
def nearestLog(x):
    if x<10000:
        return 10000
    elif x<100000:
        return 100000
    elif x<1000000:
        return 1000000
    else:
        raise("error in LogishLocator: nearestLog")
        
def convertToLogishCoordinates(Ticlocs, vmin, vmax, T):
    actualRange=vmax-vmin
    transvmax = ag.logishTransform([vmax], T)
    transvmin = ag.logishTransform([vmin], T)
    transformedRange = transvmax[0] - transvmin[0] 
    tTiclocs=[]
    for tic in ag.logishTransform(Ticlocs, T):
        ttic=tic/transformedRange*actualRange
        tTiclocs.append(ttic)
    assert len(tTiclocs)==len(Ticlocs)
    return tTiclocs

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
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if self.numticks == 'auto':
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks
        
        
        if vmin < self.T:
            tmpTicLoc=[-50000, -40000, -30000, -20000, -10000, -5000, 0, 1000, 5000]
            Ticlocs = list(set(np.clip(tmpTicLoc, nearest1k(vmin), self.T)))
        else:
            Ticlocs=np.arange(0,self.T+1, 1000)
        
        if vmax > self.T:
            tmpTicLoc=list(np.arange(self.T,10001,1000))
            tmpTicLoc.extend(np.arange(10000.0,100001,10000))
            tmpTicLoc.extend(np.arange(100000.0,500000,100000)) #[10000.0,100000.0, 200000, 300000, 1000000.0]
            Ticlocs.extend(tmpTicLoc)
            Ticlocs=list(set(Ticlocs))
        Ticlocs=convertToLogishCoordinates(Ticlocs, vmin, vmax, self.T)
        if vmax < vmin:
            vmin, vmax = vmax, vmin        
        return self.raise_if_exceeds(np.asarray(Ticlocs))

    def view_limits(self, vmin, vmax):
        'Try to choose the view limits intelligently'
        vmin, vmax = self.nonsingular(vmin, vmax)

        if rcParams['axes.autolimit_mode'] == 'round_numbers':
            if not vmin in np.arange(-10000,1000,1000):
                vmin = nearest1k(vmin)
            if not is_decade(vmax, self._base):
                vmax = decade_up(vmax, self._base)
        return vmin, vmax

    def nonsingular(self, vmin, vmax):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 1, 10  # initial range, no data plotted yet

        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmax <= 0:
            sys.stderr.write(
                "Data has no positive values, and therefore cannot be "
                "log-scaled.")
            return 1, 10

        minpos = self.axis.get_minpos()
        if not np.isfinite(minpos):
            minpos = 1e-300  # This should never take effect.
        if vmin <= 0:
            vmin = minpos
        if vmin == vmax:
            vmin = decade_down(vmin, self._base)
            vmax = decade_up(vmax, self._base)
        return vmin, vmax


class LogishFormatter(Formatter):
    """
    Base class for formatting ticks on a logish.
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
        if not round(x,0) in [0,1000,10000,100000,1000000]:
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
        tVals = ag.logishTransform([vmin, vmax, x], self._linthresh)
        # only label the decades
        fx = x/(vmax-vmin)*(tVals[1] - tVals[0])
        fx = ag.inverseLogishTransform([fx],self._linthresh)[0]
        s = self._num_to_string(fx, vmin, vmax)
        return self.fix_minus(s)
        

    def pprint_val(self, x, d):
        #If the number is at or below the set lin-cutoff (_lintrehsh)
        #Print it as an int
        #TODO: WHY DO I NEED THE +1 HERE?
        if abs(x) <= self._linthresh+1:
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