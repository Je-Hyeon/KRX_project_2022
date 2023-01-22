
import numpy as np
import pandas as pd

def zscore(df):
    df_value = df.T.values
    # result = np.zeros_like(df_value)*np.nan

    # for i in range(df_value.shape[0]):
    #     arr = df_value[i,:]
    #     result[i,:] = 

    result = np.apply_along_axis(lambda arr: (arr - np.nanmean(arr))/np.nanstd(arr), 1, df_value)

    return pd.DataFrame(result.T, index=df.index, columns=df.columns)

def rank_inner(arr):
    ranks = arr.argsort(kind='mergesort').argsort(kind='mergesort')
    ranks = np.where(ranks>=(ranks.shape[0]-np.isnan(arr).sum()), np.nan, ranks)
    return ranks/np.nanmax(ranks)

def rank(df):
    df_value = df.values.T
    # result = np.zeros_like(df_value)*np.nan

    # for i in range(df_value.shape[0]):
    #     arr = df_value[i,:]
    #     ranks = arr.argsort(kind='mergesort').argsort(kind='mergesort')
    #     ranks = np.where(ranks>=(ranks.shape[0]-np.isnan(arr).sum()), np.nan, ranks)
    #     result[i,:] = ranks/np.nanmax(ranks)

    result = np.apply_along_axis(rank_inner, 1, df_value)

    return pd.DataFrame(result.T, index=df.index, columns=df.columns)

def minmax_inner(arr):
    min = np.nanmin(arr)
    max = np.nanmax(arr)
    return (arr-min) / (max - min)

def minmax(df):
    df_value = df.values.T
    # result = np.zeros_like(df_value)*np.nan

    # for i in range(df_value.shape[0]):
    #     arr = df_value[i,:]
    #     min = np.nanmin(arr)
    #     max = np.nanmax(arr)
    #     result[i,:] = (arr-min) / (max - min)

    result = np.apply_along_axis(minmax_inner, 1, df_value)

    return pd.DataFrame(result.T, index=df.index, columns=df.columns)

def quratile_inner(arr):
    q1 = np.nanpercentile(arr, 25)
    q2 = np.nanpercentile(arr, 50)
    q3 = np.nanpercentile(arr, 75)
    return (arr-q2) / (q3 - q1)

def quratile(df):
    df_value = df.values.T
    # result = np.zeros_like(df_value)*np.nan

    # for i in range(df_value.shape[0]):
    #     arr = df_value[i,:]
    #     q1 = np.nanpercentile(arr, 25)
    #     q2 = np.nanpercentile(arr, 50)
    #     q3 = np.nanpercentile(arr, 75)
    #     result[i,:] = (arr-q2) / (q3 - q1)

    result = np.apply_along_axis(quratile_inner, 1, df_value)

    return pd.DataFrame(result.T, index=df.index, columns=df.columns)

def zero_one(df):
    # result = np.zeros_like(df_value)*np.nan

    # for i in range(df_value.shape[0]):
    #     arr = df_value[i,:]
    #     min = np.nanmin(arr)
    #     max = np.nanmax(arr)
    #     result[i,:] = (arr-min) / (max - min)

    return df.apply(lambda x: 1 if x>=1 else 0)
