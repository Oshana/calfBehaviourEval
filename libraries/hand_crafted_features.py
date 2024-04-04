import pandas as pd
import numpy as np
import antropy as ant # https://raphaelvallat.com/antropy/build/html/generated/antropy.spectral_entropy.html
from scipy.stats import kurtosis, skew

def check_array_values(arr, threshold=1e-14): # threshold value identified Experimentally
    if len(np.unique(arr)) < 2:
        return False 
    # the difference between at least two values needs to exceed the threshold to dissatisfy a flat distribution
    if np.max(arr) - np.min(arr) > threshold: 
        return True
    return False

def calculate_entropy(signal, sf=25):
    if check_array_values(signal):
        return ant.spectral_entropy(signal, sf, method='fft', normalize=True)
    else:
        '''entropy means the uncertainty. if all the values in data are the same, there is no randomness or uncertainty. Thus the entropy becomes 0. It is the same scenario when all the values are 0. But it will make sum_absAmagFFT == 0, thus giving out the division by zero error. In this scenario also it should return 0'''
        return 0

def calculate_motion_variation(data):
    return np.mean(np.abs(np.diff(data)))


def return_HC_features(data_arr, nan_allowance=0.75):
    data = data_arr[~np.isnan(data_arr)] # Remove NaN values
    
    features = []
    # size of data after dropping the NaN values needs to be atleast 75% (default) of the original data size
    if len(data) > len(data_arr) * nan_allowance:
        data_mean = np.mean(data)
        data_median = np.median(data)
        data_min = np.min(data)
        data_max = np.max(data)
        data_std = np.std(data)
        data_q1 = np.quantile(data, 0.25)
        data_q3 = np.quantile(data, 0.75)
        
        if check_array_values(data):  # to account for flat distributions
            ext_features = [kurtosis(data, nan_policy='omit'), skew(data, nan_policy='omit')]
        else:
            # flat distribution scenario
            ext_features = [-3, 0]

        entropy = calculate_entropy(data, sf=25)
        motion_variation = calculate_motion_variation(data)

        # Extend feature list
        features.extend([
            data_mean, data_median, data_min, data_max,
            data_std, data_q1, data_q3,
            entropy, motion_variation
        ] + ext_features)
            
    return features