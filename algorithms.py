# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:13:55 2023
modified on Mon Feb 27 23:55:55 2024
@author: Guoheng Qi, qigh(#)semi.ac.cn (#->@)

Reference: http://geophydog.cool/post/signal_pws_stack/
"""

import math,os,obspy,scipy,threading,time
import numpy as np
import pandas as pd
import numpy.fft as fftpack
import scipy.signal as sgn
from scipy.fftpack import rfft,irfft,fftfreq
from scipy.optimize import curve_fit
# from stockwell import st
from scipy.signal import hilbert
from multiprocessing import Process
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import font_manager,cm
from matplotlib.colors import Normalize
from matplotlib.dates import MINUTELY,AutoDateLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,FuncFormatter
from obspy import Trace,Stream,UTCDateTime,read
from obspy.signal.util import smooth
import threading,multiprocessing


plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['Times New Roman']

#============================================================================
'''
      phase stack (c)
'''
#============================================================================

def get_coh(st, v, sm = False, sl = 20):
    # '''
    # input: t, a stream;  v>=0;   sm, a bool;  sl, smooth window width
    # output: t, phase stack
    # '''
    m  = len(st)
    n  = st[0].stats.npts
    dt = st[0].stats.delta
    t  = np.arange(n) * dt
    ht = np.zeros((m, n), dtype=complex)
    c  = np.zeros(n)
    for i, tr in enumerate(st):
        ht[i] = hilbert(tr.data)
    pha = ht / abs(ht)
    for i in range(n):
        c[i] = abs( sum(pha[:, i]) )
    # Smooth the coherence if necessary.
    if sm:
        c = np.convolve(c/m, np.ones(sl)/sl, 'same') ** v
    else:
        c = ( c/m ) ** v
    return t, c

#============================================================================
'''
      phase cross-correlation (c)
'''
#============================================================================

def PCC(tr1,tr2,v,t,m):
    # '''
    # input: tr1,tr2 are traces
    #        v >=0, ���
    #        t is max delayed time 
    #        m is number of time delay
    # output: phase cross-correlation
    # '''
#    print('PCC is start!' )
    n   = tr1.stats.npts
    dt  = tr1.stats.delta
    pcc = np.zeros(m)
    ph1 = hilbert(tr1.data) / abs(hilbert(tr1.data))
    ph2 = hilbert(tr2.data) / abs(hilbert(tr2.data))

    for i in np.arange(m):
        ph0 = ph1*0
        # lag = i*t/m
        # et  = int(lag/dt)
        et = i
        ph0[:n-et] = ph2[et:]
#        print('lag time is {} is end'.format(lag))
        pcc[i] = 1/2/n * ( sum( abs(ph1+ph0)**v - abs(ph0-ph1)**v ) )
#    plt.plot(pcc)
    Pcc = Trace()
    Pcc.data = pcc
    Pcc.stats.station = tr1.stats.station
    Pcc.stats.channel = tr1.stats.channel
    Pcc.stats.location = tr1.stats.location
    Pcc.stats.starttime = tr1.stats.starttime
#    print('PCC is end!' )
    return Pcc
#============================================================================
'''
      phase-weight stack (PWS)
'''
#============================================================================

def PWS(st, v, sm=False, sl=15):
    
    print('PWS is start!' )
    m = len(st)
    n = st[0].stats.npts
#    dt = st[0].stats.delta
#    t = np.arange(n) * dt
    c = np.zeros(n, dtype=complex)
    for i, tr in enumerate(st):
        h = hilbert(tr.data)
        c += h/abs(h)
    c = abs(c/m)
    if sm:
        operator = np.ones(sl) / sl
        c = np.convolve(c, operator, 'same')
    stc = st.copy()
    stc.stack()
    tr = stc[0]
    tr.data = tr.data*c**v
    
    print('PWS is end!' )
    return tr
