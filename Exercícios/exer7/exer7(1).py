#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from scipy import stats
from numpy import loadtxt
import matplotlib.pyplot as plt
from powernoise import powernoise
from p_model import pmodel
from logistic import logistic
from henon import henon
plt.rcParams['figure.figsize'] = (10,10)

#MFDFA-Analytics-by-SKDataScience
#multifractal DFA singularity spectra - module 01
#Version 3.0 - Modified by R.R.Rosa - Dec 2018 - mfdfa_ss_m1.py
# This code implements a modification of the first-order unifractal analysis algorithm originally described in [1].
# It covers both the detrended fluctuation analysis (DFA) and the Hurst (a.k.a. R/S) analysis methods. For more details
# on the DFA and Hurst analysis methods, please refer to [2, 3].
#
# At the input, 'dx' is a time series of increments of the physical observable 'x(t)', of the length equal to an
# integer power of two greater than two (i.e. 4, 8, 16, 32, etc.), 'normType_p' is any real greater than or
# equal to one specifying the p-norm, 'isDFA' is a boolean value prescribing to use either the DFA-based algorithm or
# the standard Hurst (a.k.a. R/S) analysis, 'normType_q' is any real greater than or equal to one specifying the q-norm.
#
# At the output, 'timeMeasure' is the time measure of the data's support at different scales, 'meanDataMeasure' is
# the data measure at different scales, while 'scales' is the scales at which the data measure is computed.
#
# The conventional way of using the output values is to plot the data measure vs the scales; the time measure,
# being the inverse quantity to the scales, is computed for an alternative representation and may be ignored.
#
# The requirement to have a power-of-two data length is aimed at avoiding inaccuracies when computing the data measure
# on different time scales.
#
# REFERENCES:
# [1] D.M. Filatov, J. Stat. Phys., 165 (2016) 681-692. DOI: 10.1007/s10955-016-1641-6.
# [2] J.W. Kantelhardt, Fractal and Multifractal Time Series, available at http://arxiv.org/abs/0804.0747, 2008.
# [3] J. Feder, Fractals, Plenum Press, New York, 1988.
#
# The end user is granted perpetual permission to reproduce, adapt, and/or distribute this code, provided that
# an appropriate link is given to the original repository it was downloaded from.

#input: read your time series as a 1d vector, with size 2ˆn, named:  dx


def getHurstByUpscaling(dx, normType_p = np.inf, isDFA = 1, normType_q = 1.0):
    ## Some initialiation
    dx_len = len(dx)
    
    # We have to reserve the most major scale for shifts, so we divide the data
    # length by two. (As a result, the time measure starts from 2.0, not from
    # 1.0, see below.)
    dx_len = np.int(dx_len / 2)
    
    dx_shift = np.int(dx_len / 2)
    
    nScales = np.int(np.round(np.log2(dx_len)))    # Number of scales involved. P.S. We use 'round()' to prevent possible malcomputing of the logarithms
    j = 2 ** (np.arange(1, nScales + 1) - 1) - 1
    
    meanDataMeasure = np.zeros(nScales)
    
    ## Computing the data measure
    for ji in range(1, nScales + 1):
        # At the scale 'j(ji)' we deal with '2 * (j(ji) + 1)' elements of the data 'dx'
        dx_k_len = 2 * (j[ji - 1] + 1)
        n = np.int(dx_len / dx_k_len)
        
        dx_leftShift = np.int(dx_k_len / 2)
        dx_rightShift = np.int(dx_k_len / 2)
        
        for k in range(1, n + 1):
            # We get a portion of the data of the length '2 * (j(ji) + 1)' plus the data from the left and right boundaries
            dx_k_withShifts = dx[(k - 1) * dx_k_len + 1 + dx_shift - dx_leftShift - 1 : k * dx_k_len + dx_shift + dx_rightShift]
            
            # Then we perform free upscaling and, using the above-selected data (provided at the scale j = 0),
            # compute the velocities at the scale 'j(ji)'
            j_dx = np.convolve(dx_k_withShifts, np.ones(dx_rightShift), 'valid')
            
            # Then we compute the accelerations at the scale 'j(ji) + 1'
            r = (j_dx[1 + dx_rightShift - 1 : ] - j_dx[1 - 1 : -dx_rightShift]) / 2.0
            
            # Finally, we compute the range ...
            if (normType_p == 0):
                R = np.max(r[2 - 1 : ]) - np.min(r[2 - 1 : ])
            elif (np.isinf(normType_p)):
                R = np.max(np.abs(r[2 - 1 : ]))
            else:
                R = (np.sum(r[2 - 1 : ] ** normType_p) / len(r[2 - 1 : ])) ** (1.0 / normType_p)
            # ... and the normalisation factor ("standard deviation")
            S = np.sqrt(np.sum(np.abs(np.diff(r)) ** 2.0) / (len(r) - 1))
            if (isDFA == 1):
                S = 1.0
            
            meanDataMeasure[ji - 1] += (R / S) ** normType_q
        meanDataMeasure[ji - 1] = (meanDataMeasure[ji - 1] / n) ** (1.0 / normType_q)
    
    # We pass from the scales ('j') to the time measure; the time measure at the scale j(nScales) (the most major one)
    # is assumed to be 2.0, while it is growing when the scale is tending to j(1) (the most minor one).
    # (The scale j(nScales)'s time measure is NOT equal to 1.0, because we reserved the highest scale for shifts
    # in the very beginning of the function.)
    timeMeasure = 2.0 * dx_len / (2 * (j + 1))
    
    scales = j + 1
    
    return [timeMeasure, meanDataMeasure, scales]

#MFDFA-Analytics-by-SKDataScience
#multifractal DFA singularity spectra - module 02
#Version 3.0 - Modified by R.R.Rosa - Dec 2018 - mfdfa_ss_m2.py
# This code implements a modification of the first-order multifractal analysis algorithm. It is based on the
# corresponding unifractal analysis technique described in [1]. It computes the Lipschitz-Holder multifractal
# singularity spectrum, as well as the minimum and maximum generalised Hurst exponents [2, 3].
#
# At the input, 'dx' is a time series of increments of the physical observable 'x(t)', of the length equal to an
# integer power of two greater than two (i.e. 4, 8, 16, 32, etc.), 'normType' is any real greater than or
# equal to one specifying the p-norm, 'isDFA' is a boolean value prescribing to use either the DFA-based algorithm or
# the standard Hurst (a.k.a. R/S) analysis, 'isNormalised' is a boolean value prescribing either to normalise the
# intermediate range-to-deviation (R/S) expression or to proceed computing without normalisation.
#
# At the output, 'timeMeasure' is the time measure of the data's support at different scales, 'dataMeasure' is
# the data measure at different scales computed for each value of the variable q-norm, 'scales' is the scales at which
# the data measure is computed, 'stats' is the structure containing MF-DFA statistics, while 'q' is the values of the
# q-norm used.
#
# Similarly to unifractal analysis (see getHurstByUpscaling()), the time measure is computed merely for an alternative
# representation of the dependence 'dataMeasure(q, scales) ~ scales ^ -tau(q)'.
#
# REFERENCES:
# [1] D.M. Filatov, J. Stat. Phys., 165 (2016) 681-692. DOI: 10.1007/s10955-016-1641-6.
# [2] J.W. Kantelhardt, Fractal and Multifractal Time Series, available at http://arxiv.org/abs/0804.0747, 2008.
# [3] J. Feder, Fractals, Plenum Press, New York, 1988.
#
# The end user is granted perpetual permission to reproduce, adapt, and/or distribute this code, provided that
# an appropriate link is given to the original repository it was downloaded from.



def getMSSByUpscaling(dx, normType = np.inf, isDFA = 1, isNormalised = 1):
    ## Some initialiation
    aux_eps = np.finfo(float).eps
    
    # We prepare an array of values of the variable q-norm
    aux = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.0001, 0.0, 0.0001, 0.5, 0.9999, 1.0, 1.0001, 2.0, 4.0, 8.0, 16.0, 32.0]
    nq = len(aux)
    
    q = np.zeros((nq, 1))
    q[:, 1 - 1] = aux
    
    dx_len = len(dx)
    
    # We have to reserve the most major scale for shifts, so we divide the data
    # length by two. (As a result, the time measure starts from 2.0, not from
    # 1.0, see below.)
    dx_len = np.int(dx_len / 2)
    
    dx_shift = np.int(dx_len / 2)
    
    nScales = np.int(np.round(np.log2(dx_len)))    # Number of scales involved. P.S. We use 'round()' to prevent possible malcomputing of the logarithms
    j = 2 ** (np.arange(1, nScales + 1) - 1) - 1
    
    dataMeasure = np.zeros((nq, nScales))
    
    ## Computing the data measures in different q-norms
    for ji in range(1, nScales + 1):
        # At the scale 'j(ji)' we deal with '2 * (j(ji) + 1)' elements of the data 'dx'
        dx_k_len = 2 * (j[ji - 1] + 1)
        n = np.int(dx_len / dx_k_len)
        
        dx_leftShift = np.int(dx_k_len / 2)
        dx_rightShift = np.int(dx_k_len / 2)
        
        R = np.zeros(n)
        S = np.ones(n)
        for k in range(1, n + 1):
            # We get a portion of the data of the length '2 * (j(ji) + 1)' plus the data from the left and right boundaries
            dx_k_withShifts = dx[(k - 1) * dx_k_len + 1 + dx_shift - dx_leftShift - 1 : k * dx_k_len + dx_shift + dx_rightShift]
            
            # Then we perform free upscaling and, using the above-selected data (provided at the scale j = 0),
            # compute the velocities at the scale 'j(ji)'
            j_dx = np.convolve(dx_k_withShifts, np.ones(dx_rightShift), 'valid')
            
            # Then we compute the accelerations at the scale 'j(ji) + 1'
            r = (j_dx[1 + dx_rightShift - 1 : ] - j_dx[1 - 1 : -dx_rightShift]) / 2.0
            
            # Finally we compute the range ...
            if (normType == 0):
                R[k - 1] = np.max(r[2 - 1 : ]) - np.min(r[2 - 1 : ])
            elif (np.isinf(normType)):
                R[k - 1] = np.max(np.abs(r[2 - 1 : ]))
            else:
                R[k - 1] = (np.sum(r[2 - 1 : ] ** normType) / len(r[2 - 1 : ])) ** (1.0 / normType)
            # ... and the normalisation factor ("standard deviation")
            if (isDFA == 0):
                S[k - 1] = np.sqrt(np.sum(np.abs(np.diff(r)) ** 2.0) / (len(r) - 1))
    
        if (isNormalised == 1):      # Then we either normalise the R / S values, treating them as probabilities ...
            p = np.divide(R, S) / np.sum(np.divide(R, S))
        else:                        # ... or leave them unnormalised ...
            p = np.divide(R, S)
          # ... and compute the measures in the q-norms
        for k in range(1, n + 1):
            # This 'if' is needed to prevent measure blow-ups with negative values of 'q' when the probability is close to zero
            if (p[k - 1] < 1000.0 * aux_eps):
                continue
            
            dataMeasure[:, ji - 1] = dataMeasure[:, ji - 1] + np.power(p[k - 1], q[:, 1 - 1])

    # We pass from the scales ('j') to the time measure; the time measure at the scale j(nScales) (the most major one)
    # is assumed to be 2.0, while it is growing when the scale is tending to j(1) (the most minor one).
    # (The scale j(nScales)'s time measure is NOT equal to 1.0, because we reserved the highest scale for shifts
    # in the very beginning of the function.)
    timeMeasure = 2.0 * dx_len / (2 * (j + 1))
    
    scales = j + 1
    
    ## Determining the exponents 'tau' from 'dataMeasure(q, timeMeasure) ~ timeMeasure ^ tau(q)'
    tau = np.zeros((nq, 1))
    log10tm = np.log10(timeMeasure)
    log10dm = np.log10(dataMeasure)
    log10tm_mean = np.mean(log10tm)
    
    # For each value of the q-norm we compute the mean 'tau' over all the scales
    for qi in range(1, nq + 1):
        tau[qi - 1, 1 - 1] = np.sum(np.multiply(log10tm, (log10dm[qi - 1, :] - np.mean(log10dm[qi - 1, :])))) / np.sum(np.multiply(log10tm, (log10tm - log10tm_mean)))

    ## Finally, we only have to pass from 'tau(q)' to its conjugate function 'f(alpha)'
    # In doing so, first we find the Lipschitz-Holder exponents 'alpha' (represented by the variable 'LH') ...
    aux_top = (tau[2 - 1] - tau[1 - 1]) / (q[2 - 1] - q[1 - 1])
    aux_middle = np.divide(tau[3 - 1 : , 1 - 1] - tau[1 - 1 : -1 - 1, 1 - 1], q[3 - 1 : , 1 - 1] - q[1 - 1 : -1 - 1, 1 - 1])
    aux_bottom = (tau[-1] - tau[-1 - 1]) / (q[-1] - q[-1 - 1])
    LH = np.zeros((nq, 1))
    LH[:, 1 - 1] = -np.concatenate((aux_top, aux_middle, aux_bottom))
    # ... and then compute the conjugate function 'f(alpha)' itself
    f = np.multiply(LH, q) + tau

    ## The last preparations
    # We determine the minimum and maximum values of 'alpha' ...
    LH_min = LH[-1, 1 - 1]
    LH_max = LH[1 - 1, 1 - 1]
    # ... and find the minimum and maximum values of another multifractal characteristic, the so-called
    # generalised Hurst (or DFA) exponent 'h'. (These parameters are computed according to [2, p. 27].)
    h_min = -(1.0 + tau[-1, 1 - 1]) / q[-1, 1 - 1]
    h_max = -(1.0 + tau[1 - 1, 1 - 1]) / q[1 - 1, 1 - 1]
    
    stats = {'tau':       tau,
        'LH':        LH,
            'f':         f,
                'LH_min':    LH_min,
                    'LH_max':    LH_max,
                        'h_min':     h_min,
                            'h_max':     h_max}
    
    return [timeMeasure, dataMeasure, scales, stats, q]


#MFDFA-Analytics-by-SKDataScience
#multifractal DFA singularity spectra - module 03
#Version 3.0 - Modified by R.R.Rosa - Dec 2018 - mfdfa_ss_m3.py
# This function determines the optimal linear approximations of the data measure using two segments and returns
# the index of the corresponding boundary scale (a.k.a. crossover), the boundary scale itself, as well as the
# unifractal characteristics at the major and minor scales. For examples of using crossovers, see [1, 2].
#
# At the input, 'timeMeasure' is a time measure at different scales, while 'dataMeasure' is a data measure at the same
# scales.
#
# At the output, 'bScale' is the boundary scale, or crossover, separating the major and minor scales, 'bDM' is the
# data measure at the boundary scale, 'bsIndex' is the crossover's index with respect to the time measure, 'HMajor' is
# the unifractal dimension at the major scales, 'HMinor' is the unifractal dimension at the minor scales.
#
# REFERENCES:
# [1] D.M. Filatov, J. Stat. Phys., 165 (2016) 681-692. DOI: 10.1007/s10955-016-1641-6.
# [2] C.-K. Peng, S. Havlin, H.E. Stanley and A.L. Goldberger, Chaos, 5 (1995) 82–87. DOI: 10.1063/1.166141.
#
# The end user is granted perpetual permission to reproduce, adapt, and/or distribute this code, provided that
# an appropriate link is given to the original repository it was downloaded from.


def getScalingExponents(timeMeasure, dataMeasure):
    ## Initialisation
    nScales = len(timeMeasure)
    
    log10tm = np.log10(timeMeasure)
    log10dm = np.log10(dataMeasure)
    
    res = 1.0e+07
    bsIndex = nScales
    
    ## Computing
    # We find linear approximations for major and minor subsets of the data measure and determine the index of the
    # boundary scale at which the approximations are optimal in the sense of best fitting to the data measure
    for i in range(3, nScales - 2 + 1):
        # Major 'i' scales are approximated by the function 'k * x + b' ...
        curr_log10tm = log10tm[nScales - i + 1 - 1 : nScales]
        curr_log10dm = log10dm[nScales - i + 1 - 1 : nScales]
        detA = i * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
        detK = i * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
        detB = np.sum(curr_log10dm) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) * np.sum(np.multiply(curr_log10tm, curr_log10dm))
        k = detK / detA
        b = detB / detA
        # ... and the maximum residual is computed
        resMajor = max(np.abs(k * curr_log10tm + b - curr_log10dm))
        
        # Minor 'nScales - i + 1' scales are approximated by the function 'k * x + b' ...
        curr_log10tm = log10tm[1 - 1 : nScales - i + 1]
        curr_log10dm = log10dm[1 - 1 : nScales - i + 1]
        detA = (nScales - i + 1) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
        detK = (nScales - i + 1) * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
        detB = np.sum(curr_log10dm) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) * np.sum(np.multiply(curr_log10tm, curr_log10dm))
        k = detK / detA
        b = detB / detA
        # ... and the maximum residual is computed
        resMinor = max(np.abs(k * curr_log10tm + b - curr_log10dm))
        
        if (resMajor ** 2.0 + resMinor ** 2.0 < res):
            res = resMajor ** 2.0 + resMinor ** 2.0
            bsIndex = i

    # Now we determine the boundary scale and the boundary scale's data measure, ...
    bScale = 2.0 * timeMeasure[1 - 1] / timeMeasure[nScales - bsIndex + 1 - 1] / 2.0
    bDM = dataMeasure[nScales - bsIndex + 1 - 1]
    # ... as well as compute the unifractal dimensions using the boundary scale's index:
    # at the major 'bsIndex' scales ...
    curr_log10tm = log10tm[nScales - bsIndex + 1 - 1 : nScales]
    curr_log10dm = log10dm[nScales - bsIndex + 1 - 1 : nScales]
    detA = bsIndex * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
    detK = bsIndex * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
    DMajor = detK / detA
    HMajor = -DMajor
    # ... and at the minor 'nScales - bsIndex + 1' scales
    curr_log10tm = log10tm[1 - 1 : nScales - bsIndex + 1]
    curr_log10dm = log10dm[1 - 1 : nScales - bsIndex + 1]
    detA = (nScales - bsIndex + 1) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
    detK = (nScales - bsIndex + 1) * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
    DMinor = detK / detA
    HMinor = -DMinor
    
    return [bScale, bDM, bsIndex, HMajor, HMinor]



def k_means(data):

    # Descobrindo o k ótimo
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=10)
    visualizer.fit(data)
    plt.close()

    # Aplicando o k-means
    n_clusters = visualizer.elbow_value_ #Número ótimo de clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    data['K_classes'] = kmeans.labels_
    centroides = kmeans.cluster_centers_

    #Plotando as relações a pares de cada medida estatística (seaborn)
    colors = sns.color_palette('tab10',n_clusters)

    try:
        g = sns.pairplot(data, 'K_classes', palette=colors, height=5)
        g.fig.suptitle('\n\nEspaço de parâmetros: S² x '+u'\u03A8' , y=1.08)
        plt.show()
    except ValueError:
        print('\nOs dados não apresentam variância')
    except RuntimeError:
        print('\nOs dados não apresentam variância')

    for classe in pd.unique(data['K_classes']):
        print(f'\n\n****************** Sinais na classe {classe} ******************')
        print(data.loc[data['K_classes']==classe])

            

def colname(n,name):
    """
    Retorna uma lista com as strings dos valores de N e sinal atuais
    """

    col = []
    for i in range(n):
        col.append(name+'_'+str(i))

    return col  


#MFDFA-Analytics-by-SKDataScience
#multifractal DFA singularity spectra - module 04
#Version 3.0 - Modified by R.R.Rosa - Dec 2018 - mfdfa_ss_m4.py
#This module is the entry point for testing the modified first-order uni- and multifractal DFA methods.
#The initial dataset is a time series of size 2ˆn (tseries.txt)

## Loading data
n_iter = 10
n_sinais = 2**13
assimetria_ = []
psi = []

# Cria 10 sinais referentes às séries do dataset_signal
S1 = []
S2 = []
S3 = []
S4 = []
S5 = []
S6 = []
S7 = []
S8 = []

for n in range(n_iter):
    #série temporal estocástica
    res = n_sinais/12
    S1.append((np.random.randn(n_sinais) * np.sqrt(res) * np.sqrt(1 / n_sinais)).cumsum())

    #ruído branco
    S2.append(powernoise(0,n_sinais))

    #ruído rosa
    S3.append(powernoise(1,n_sinais))

    #ruído vermelho
    S4.append(powernoise(2,n_sinais))

    #sinal endógeno
    p_endo = np.random.uniform(0.32,0.42)
    x, y = (pmodel(n_sinais,p_endo,0.4))
    S5.append(y)

    #sinal exógeno
    p_exo = np.random.uniform(0.18,0.28)
    x, y = (pmodel(n_sinais,p_exo,0.7))
    S6.append(y)

    #mapa logístico
    rho = np.random.uniform(3.81,4.00)
    S7.append(logistic(rho,0.001,n_sinais))

    #mapa henon
    a = np.random.uniform(1.35,1.42)
    S8.append(henon(a,0.3,0.1,n_sinais))

tmp1 = pd.DataFrame(np.transpose(S1),columns=colname(n_iter,'S1'))
tmp2 = pd.DataFrame(np.transpose(S2),columns=colname(n_iter,'S2'))
tmp3 = pd.DataFrame(np.transpose(S3),columns=colname(n_iter,'S3'))
tmp4 = pd.DataFrame(np.transpose(S4),columns=colname(n_iter,'S4'))
tmp5 = pd.DataFrame(np.transpose(S5),columns=colname(n_iter,'S5'))
tmp6 = pd.DataFrame(np.transpose(S6),columns=colname(n_iter,'S6'))
tmp7 = pd.DataFrame(np.transpose(S7),columns=colname(n_iter,'S7'))
tmp8 = pd.DataFrame(np.transpose(S8),columns=colname(n_iter,'S8'))

df = pd.concat([tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8],axis=1)
 # Dataframe com os momentos estatísticos dos sinas acima
df_stats = pd.DataFrame(np.transpose([stats.skew(df)**2]),columns=['skewness²']).set_index(df.columns)

for col in df.columns:
    name = col
    dx = df[col].values
    size=8192
    dx = dx[1 - 1 : 8192]               # We take the first 8192 samples

    ## Computing
    # Modified first-order DFA
    [timeMeasure, meanDataMeasure, scales] = getHurstByUpscaling(dx)                    # Set of parameters No. 1
    #[timeMeasure, meanDataMeasure, scales] = getHurstByUpscaling(dx, 3.0, 0, 2.0)       # Set of parameters No. 2

    [bScale, bDM, bsIndex, HMajor, HMinor] = getScalingExponents(timeMeasure, meanDataMeasure)

    # Modified first-order MF-DFA
    [_, dataMeasure, _, stats, q] = getMSSByUpscaling(dx, isNormalised = 1)

    # Asymmetry of the alpha plot
    index = max(range(len(stats['f'])), key=stats['f'].__getitem__)
    alfa_zero = stats['LH'][index]
    asymmetry = (alfa_zero - min(stats['LH']))/(max(stats['LH']) - alfa_zero)
    A = round(float(''.join([str(x) for x in asymmetry])),3)
    
    # Getting the psi index
    psi.append((stats['LH_max'] - stats['LH_min'])/stats['LH_max'])
    
    #Plot the first signal of each serie
    signal_iter = name.split('_')[1]
    if signal_iter == '0': 

        ## Output
        print('\n\n\n======================================== Serie: '+name+' ========================================')
        # Modified first-order DFA
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.loglog(timeMeasure, meanDataMeasure, 'ko-')
        plt.xlabel(r'$\mu(t)$')
        plt.ylabel(r'$\mu(\Delta x)$')
        plt.grid('on', which = 'minor')
        plt.title('Modified First-Order DFA of a Multifractal Noise')

        plt.subplot(2, 1, 2)
        plt.loglog(scales, meanDataMeasure, 'ko-')
        plt.loglog(bScale, bDM, 'ro')
        plt.xlabel(r'$j$')
        plt.ylabel(r'$\mu(\Delta x)$')
        plt.grid('on', which = 'minor')

        # Modified first-order MF-DFA
        print('alpha_min = %g, alpha_max = %g, dalpha = %g' % (stats['LH_min'], stats['LH_max'], stats['LH_max'] - stats['LH_min']))
        print('h_min = %g, h_max = %g, dh = %g\n' % (stats['h_min'], stats['h_max'], stats['h_max'] - stats['h_min']))
        print(u'\u03A8: ',(stats['LH_max'] - stats['LH_min'])/stats['LH_max'])

        plt.figure()
        nq = np.int(len(q))
        leg_txt = []
        for qi in range(1, nq + 1):
            llh = plt.loglog(scales, dataMeasure[qi - 1, :], 'o-')
            leg_txt.append('tau = %g (q = %g)' % (stats['tau'][qi - 1], q[qi - 1]))
        plt.xlabel(r'$j$')
        plt.ylabel(r'$\mu(\Delta x, q)$')
        plt.grid('on', which = 'minor')
        plt.title('Modified First-Order MF-DFA of a Multifractal Noise')
        plt.legend(leg_txt)

        plt.figure()

        #plt.subplot(2, 1, 1)
        plt.plot(q, stats['tau'], 'ko-')
        plt.xlabel(r'$q$')
        plt.ylabel(r'$\tau(q)$')
        plt.grid('on', which = 'major')
        plt.title('Statistics of Modified First-Order MF-DFA of a Multifractal Noise')

        plt.figure()

        if A < 1:
            title = 'Right skewed spectrum (smaller amplitude fluctuations) - A: '+str(A)
        elif A == 1:
            title = 'Symmetric spectrum - A: '+str(A)
        elif A > 1:
            title = 'Left skewed spectrum (larger amplitude fluctuations) - A: '+str(A)

        #plt.subplot(2, 1, 2)
        plt.plot(stats['LH'], stats['f'], 'ko-')
        plt.title(title)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$f(\alpha)$')
        plt.grid('on', which = 'major')


        plt.show()
        
df_stats[u'\u03A8'] = psi
k_means(df_stats)


# In[ ]:




