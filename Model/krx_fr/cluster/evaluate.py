import numpy as np
import pandas as pd
from krx_fr.cluster.optimize import Optimizer
from krx_fr.cluster.kmeans import MyKmeans


def run_iter_kmeans(num_k, max_sample, raw_data):
    model_save = {}
    for time, dataframe in raw_data.items():
        cluster_model = MyKmeans(dataframe)
        cluster_model.set_params()
        cluster_model.find_optimal_initp(num_of_cluster= num_k, max_sample=max_sample, optimize_method="silhouette")
        k_mean_dict = cluster_model.run_kmean(num_of_cluster=num_k)
        model_save[time] = k_mean_dict

    return model_save

def eval_cluster_size(model_save:dict):
    return_df = pd.DataFrame()
    for t, model_dict in model_save.items():
        a = pd.Series(model_save[t]["model"].labels_).value_counts()
        a = pd.DataFrame(a)
        a.columns = [t]
        return_df = pd.concat([return_df, a], axis=1)
        return_df.sort_index(inplace=True)
    return_df.loc["기업수",:] = return_df.sum(axis=0)
    return return_df
    

def eval_cluster_result(model_save:dict, raw_data:dict, label_data:dict, freq, return_best_cluster=True):
    freq = freq + "S"
    date_range = pd.date_range("2012-01-01","2021-01-01", freq=freq)
    return_df1 = pd.DataFrame()
    return_df2 = pd.DataFrame()

    for date in date_range:
        date = str(date)[:7]
        prob, original_prob = Optimizer.eval(model_save[date]["model"], raw_data[date], label_data[date])
        df = pd.DataFrame(prob.values(), prob.keys(), columns=[date])
        df2 = pd.DataFrame(original_prob.values(), original_prob.keys(), columns=[date])
        return_df1 = pd.concat([return_df1,df], axis=1)
        return_df2 = pd.concat([return_df2, df2], axis=1)
    if return_best_cluster:
        return_df1.loc["포착군집", :] = np.argmax(return_df1.values, axis=0)
        return_df2.loc["포착군집", :] = np.argmax(return_df2.values, axis=0)

    return return_df1, return_df2


def eval_cluster_distance(prob_df, raw_data, model_save):
    ser = prob_df.loc["포착군집"].copy()
    tmp_dict = {}
    return_dict = {}
    for t,danger_cluster in ser.iteritems(): #포착군집을 반복한다
        t = str(t)
        num = int(danger_cluster)
        cluster_model = MyKmeans(raw_data[t])
        distance = cluster_model.distance_decomposition(model_save[t]["model"])[num] #최대확률 클러스터와 타 클러스터의 거리
        tmp_dict[t] = distance # 시기:거리

    for t, distance_matrix in tmp_dict.items():
        distance_sum = np.zeros(raw_data[t].columns.shape, dtype='float64')
        for i in range(len(distance_matrix)):
            slice = distance_matrix[i]
            if len(slice) ==0:
                continue
            distance_sum += slice
        return_dict[t] = distance_sum
        col = raw_data[t].columns
    
    return_df = pd.DataFrame(return_dict).T
    return_df.columns = col
    return return_df.sort_values("2013-01", axis=1, ascending=False)


def calculate_match_probability(eval_label1, eval_label2, eval_label3:None):
    if eval_label3 is not None:
        idx = eval_label1.columns
        data = np.c_[eval_label1.iloc[-1,:].values, eval_label2.iloc[-1,:].values, eval_label3.iloc[-1,:].values]
        df = pd.DataFrame(data=data, index=idx, columns=["라벨1","라벨2","라벨3"])
        con1 = (df["라벨1"] == df["라벨2"])
        con2 = (df["라벨3"] == df["라벨1"])
        return np.mean(con1==con2)

    else:
        idx = eval_label1.columns
        data = np.c_[eval_label1.iloc[-1,:].values, eval_label2.iloc[-1,:].values]
        df = pd.DataFrame(data=data, index=idx, columns=["라벨1","라벨2"])
        con1 = (df["라벨1"] == df["라벨2"])
        return np.mean(con1)
    

def calculate_catch_probability(eval_label1, eval_label2, eval_label3:None):
    if eval_label3 is not None:
        pro1 = eval_label1.iloc[0:-1,:].max(axis=0).mean()
        pro2 = eval_label2.iloc[0:-1,:].max(axis=0).mean()
        pro3 = eval_label3.iloc[0:-1,:].max(axis=0).mean()
        return np.c_[pro1,pro2,pro3].mean()
    else:
        pro1 = eval_label1.iloc[0:-1,:].max(axis=0).mean()
        pro2 = eval_label2.iloc[0:-1,:].max(axis=0).mean()
        return np.c_[pro1,pro2].mean()


def predict_cluster_result(model_save, raw_data, label_data,criterion_period:str, freq):
    if freq == 'Y':
        plus = 12
    elif freq == "Q":
        plus = 3

    date_idx = pd.date_range(criterion_period, "2023-01",freq=freq+'S')
    return_df = pd.DataFrame()
    return_df2 = pd.DataFrame()
    try:
        for date in date_idx:
            period = pd.Period(str(date)[:7]) + plus
            date = str(date)[:7]
            prob, original_prob = Optimizer.eval(model_save[date]["model"], raw_data[str(period)], label_data[str(period)])
            df = pd.DataFrame(prob.values(), prob.keys(), columns=[str(period)+"예측"])
            df2 = pd.DataFrame(original_prob.values(), original_prob.keys(), columns=[str(period)+"예측"])
            return_df = pd.concat([return_df,df], axis=1)
            return_df2 = pd.concat([return_df2,df2], axis=1)
    except:
        return return_df.dropna(axis=1), return_df2.dropna(axis=1)

def cluster_of_cluster_model(model_save,raw_data,eval_label, num_k, max_sample):
    return_model_save = {}
    return_masked_data = {}

    for t, num in eval_label.iloc[-1,:].iteritems():
        num = int(num)
        mask = model_save[t]["model"].labels_ == num
        masked_data = raw_data[t][mask]

        cluster_model = MyKmeans(masked_data)
        cluster_model.set_params()
        cluster_model.find_optimal_initp(num_of_cluster= num_k, max_sample=max_sample, optimize_method="silhouette")
        k_mean_dict = cluster_model.run_kmean(num_of_cluster=num_k)

        return_masked_data[t] = masked_data
        return_model_save[t] = k_mean_dict

    return return_model_save, masked_data

def compare_predict_prob(predict_cluster_result, eval_cluster_result):
    '''predict_cluster_result의 리턴과 eval_cluster_result의 리턴을 받습니다
    -> 실제값과 확률값을 비교해서 보여줍니다
    '''
    max_list = [] # 예측값이 담김
    min_list = []
    l1 = [] # 실제값이 담김
    l2 = [] 

    for i in range(len(eval_cluster_result.T)): # 실제값을 반복돌림
        if i == predict_cluster_result.T.shape[0]:
            continue

        max_idx = np.argmax(eval_cluster_result.T.iloc[i,:-1])
        min_idx = np.argmin(eval_cluster_result.T.iloc[i,:-1])
        max_predict = predict_cluster_result.T.iloc[i, max_idx]
        min_predict = predict_cluster_result.T.iloc[i, min_idx]

        max_idxn = np.argmax(eval_cluster_result.T.iloc[i+1,:-1])
        min_idxn = np.argmin(eval_cluster_result.T.iloc[i+1,:-1])
        real_max = eval_cluster_result.T.iloc[i+1, max_idxn]
        real_min = eval_cluster_result.T.iloc[i+1, min_idxn]
        max_list.append(max_predict)
        min_list.append(min_predict)
        l1.append(real_max)
        l2.append(real_min)
    df = pd.DataFrame(np.c_[max_list,l1,min_list,l2], columns=["예측max","실제max","예측min","실제min"], index=eval_cluster_result.T.index[1:])
    df["max차"] = df["예측max"] - df["실제max"]
    return df


def compare_predict_prob_q(predict_cluster_result, eval_cluster_result):
    '''predict_cluster_result의 리턴과 eval_cluster_result의 리턴을 받습니다
    -> 실제값과 확률값을 비교해서 보여줍니다
    '''
    max_list = [] # 예측값이 담김
    min_list = []
    l1 = [] # 실제값이 담김
    l2 = [] 

    for i in range(len(eval_cluster_result.T)): # 실제값을 반복돌림
        if i == predict_cluster_result.T.shape[0] -3:
            break

        max_idx = np.argmax(eval_cluster_result.T.iloc[i,:-1])
        min_idx = np.argmin(eval_cluster_result.T.iloc[i,:-1])
        max_predict = predict_cluster_result.T.iloc[i, max_idx]
        min_predict = predict_cluster_result.T.iloc[i, min_idx]

        max_idxn = np.argmax(eval_cluster_result.T.iloc[i+1,:-1])
        min_idxn = np.argmin(eval_cluster_result.T.iloc[i+1,:-1])
        real_max = eval_cluster_result.T.iloc[i+1, max_idxn]
        real_min = eval_cluster_result.T.iloc[i+1, min_idxn]

        max_list.append(max_predict)
        min_list.append(min_predict)
        l1.append(real_max)
        l2.append(real_min)
    df = pd.DataFrame(np.c_[max_list,l1,min_list,l2], columns=["예측max","실제max","예측min","실제min"], index=eval_cluster_result.T.index[1:])
    df["max차"] = df["예측max"] - df["실제max"]
    return df


def see_cluster_info(model_save, eval_cluster, unscaled_data):
    '''eval_cluster -> eval_cluster_result의 결과를 받는다'''
    ar = eval_cluster.iloc[:-1,:].values
    cluster = np.apply_along_axis(np.argmax, axis=0, arr=ar)
    return_dict = {}
    for (t, model_dict), cluster_no in zip(model_save.items(), cluster):
        mask = (model_dict["model"].labels_ == cluster_no)
        masked_data = unscaled_data[t].loc[mask,:] 
        mean = masked_data.mean(axis=0)
        return_dict[t] =  mean
    return return_dict