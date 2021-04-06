#!/usr/bin/env python3
#cython: language_level=3
# -*- coding: utf-8 -*-
#
#			~~aliGater~~
#	(semi)automated gating software
#
#               /^^\
#   /^^\_______/0  \_
#  (                 `~+++,,_________,,++~^^^^^^^
#..V^V^V^V^V^V^\.................................
#
#
#	Parsing flow data with fcsparser from Eugene Yurtsevs FlowCytometryTools (very slightly modified)
#	Check out his excellent toolkit for flow cytometry analysis: 
#	http://eyurtsev.github.io/FlowCytometryTools/
#
#	Björn Nilsson & Ludvig Ekdahl 2016~
#	https://www.med.lu.se/labmed/hematologi_och_transfusionsmedicin/forskning/bjoern_nilsson
#   Distributed under the MIT License

import numpy as np

from cpython cimport bool
from libc.math cimport log10, log, exp

#For cython optimisations with boundscheck, wraparound etc
cimport cython
#For cython optimization with numpy arrays
cimport numpy as np

#Declare some types for cython
ctypedef np.float64_t dtype_t
ctypedef np.int32_t itype_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def __vectorlogicleTransform(np.ndarray[dtype_t, ndim=1, mode="c"] a, double linCutOff):
    
    cdef int a_idx = 0
    cdef int a_size = len(a)
    cdef np.ndarray[dtype_t, ndim=1, mode="c"] tA = np.empty_like(a).astype(float)
    while a_idx < a_size:
        if a[a_idx] >= linCutOff:
           tA[a_idx] = log(10 * a[a_idx] / linCutOff)/log(10)
        else:
            tA[a_idx] = (a[a_idx]/linCutOff + log(10.0) - 1)/log(10)
        a_idx+=1
    
    return tA


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def __vectorInverselogicleTransform(np.ndarray[dtype_t, ndim=1, mode="c"] a, double linCutOff):
    
    cdef int a_idx = 0
    cdef int a_size = len(a)
    cdef np.ndarray[dtype_t, ndim=1, mode="c"] invA = np.empty_like(a).astype(float)
    
    while a_idx < a_size:
        if a[a_idx] >= 1.0: #transformed linCutOff, always 1.0 at T; np.log(10 * linCutOff / linCutOff)/np.log(10) -> np.log(10)/np.log(10) = 1 
            invA[a_idx] = linCutOff*exp(log(10)*a[a_idx])/10
            #invA[a_idx]= (np.exp(a[a_idx])+10)*linCutOff/10
        else:
            invA[a_idx] = linCutOff*(log(10.0)*a[a_idx] - log(10.0) + 1)
        a_idx+=1
    
    return invA


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def __vectorBilogTransform(np.ndarray[dtype_t, ndim=1, mode="c"] a, double T):
    
    cdef int a_idx = 0
    cdef int a_size = len(a)
    cdef np.ndarray[dtype_t, ndim=1, mode="c"] tA = np.empty_like(a).astype(float)
    
    while a_idx < a_size:
        if a[a_idx] >= T:
           tA[a_idx] = log(10 * a[a_idx] / T)/log(10)
        elif a[a_idx] < T and a[a_idx] > -T:
            tA[a_idx] = (a[a_idx]/T  + log(10) - 1) / log(10)
        else:
            tA[a_idx] = -log(10 * abs(a[a_idx]) / T) / log(10)+1.13141103619349642 #This shift ensures that the transformed coordinates are continous, important for bins and plotting
        a_idx+=1
        
    return tA

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def __vectorInverseBilogTransform(np.ndarray[dtype_t, ndim=1, mode="c"] a, double T):
    
    cdef int a_idx = 0
    cdef int a_size = len(a)
    cdef np.ndarray[dtype_t, ndim=1, mode="c"] invA = np.empty_like(a).astype(float)    
    cdef float tmpX
    
    while a_idx < a_size:
        if a[a_idx] >= 1.0: #transformed linCutOff, always 1.0 at T; np.log(10 * linCutOff / linCutOff)/np.log(10) -> np.log(10)/np.log(10) = 1 
            invA[a_idx] = T*exp(log(10)*a[a_idx])/10
        elif a[a_idx] <= 0.13141103619349642: #This is (np.log(10)-2)/np.log(10) I.e. the linear scale value at X=-T
            tmpX=a[a_idx]-1.13141103619349642 #This shift ensures that the transformed coordinates are continous, important for bins and plotting
            invA[a_idx] = -T*exp(log(10)*-tmpX)/10
        else:
            invA[a_idx] = T * (log(10)*a[a_idx] - log(10) + 1)
        a_idx+=1
    return invA    