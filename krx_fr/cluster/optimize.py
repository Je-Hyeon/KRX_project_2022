
import random
from itertools import combinations

import pandas as pd
import numpy as np

from tqdm import tqdm

from .kmeans import MyKmeans
from ..preprocess.tools import dict_data_drop

class Optimizer:

    def __init__(self, number_of_cluster=False, init_point=False, variable=False):
        self.__isOptimize_number_of_cluster = number_of_cluster
        self.__isOptimize_init_point = init_point
        self.__isOptimize_variable = variable

    def set_input(self, raw_data, number_of_cluster=None, init_point=None):
        if not self.__isOptimize_number_of_cluster:
            if number_of_cluster == None:
                raise ValueError("Please set the number_of_cluster")
            elif raw_data.keys() != number_of_cluster.keys():
                raise ValueError("keys for raw_data and number_of_cluster do not match")
        
        if not self.__isOptimize_init_point:
            if init_point == None:
                raise ValueError("Please set the init_point")
            elif raw_data.keys() != init_point.keys():
                raise ValueError("keys for raw_data and init_point do not match")

        self.__raw_data = raw_data
        self.__number_of_cluster = number_of_cluster
        self.__init_point = init_point
    
    def set_optimize_k(self, max_k, max_sample=500):
        """Init_point optimization 파라미터 설정
        
        Args:
            max_sample (int): 반복할 시뮬레이션 횟수
        """
        self.__optimize_k_max_k = max_k
        self.__optimize_k_max_sample = max_sample

    def set_optimize_initp(self, max_sample=500, optimize_method="inertia"):
        """Init_point optimization 파라미터 설정
        
        Args:
            max_sample (int): 반복할 시뮬레이션 횟수
            optimize_method ("intertia", "inter_std", "silhouette"): 어떤 방식으로 시작점을 결정할지
        """
        self.__optimize_initp_max_sample = max_sample
        self.__optimize_initp_method = optimize_method

    def set_kmeans_params(self, max_iter=500, tol=1e-4):
        """Kmeans 파라미터 설정
        
        Args:
            max_iter (int): 알고리즘의 최대 반복 횟수
            tol (float): loss가 tol 이하일 경우 알고리즘 반복 중단
        """
        self.__max_iter = max_iter
        self.__tol = tol
    
    def run(self, labels, random_seed=0, train_data_ratio=0.8):
        if self.__raw_data.keys() != labels.keys():
            raise ValueError("keys for raw_data and labels do not match")
    
        initial_setting = {}
        for each_data in tqdm(self.__raw_data.keys(), desc="Optimizing 'k' and 'init_point'..."):
            tmp_model = MyKmeans(self.__raw_data[each_data])
            tmp_model.set_params(self.__max_iter, self.__tol)
            k = self.__number_of_cluster[each_data]
            init_point = self.__init_point[each_data]

            if self.__isOptimize_number_of_cluster:
                tmp_model.find_optimal_k(self.__optimize_k_max_k, self.__optimize_k_max_sample, "silhouette")
                k = None
            if self.__isOptimize_init_point:
                tmp_model.find_optimal_initp(k, self.__optimize_initp_max_sample, self.__optimize_initp_method)
                init_point = None
            
            initial_setting[each_data] = [k, init_point]

        if self.__isOptimize_variable:
            random.seed(random_seed)
            train_data_list = sorted(random.sample(list(self.__raw_data.keys()), int(len(self.__raw_data.keys())*train_data_ratio)))

            result_train_data_accuracy = {}

            combinations_cnt = 0
            currently_droped = []
            while True:
                tmp_combinations = list(combinations(self.__raw_data[each_data].columns, combinations_cnt))
                tmp_combinations = [list(tmp) for tmp in tmp_combinations if len((set(tmp)) - set(currently_droped)) != len(tmp)]

                accuracy_std_list = []
                for combination in tqdm(tmp_combinations, desc="Optimizing variable..."):
                    for train_data in train_data_list:
                        tmp_model = MyKmeans(self.__raw_data[each_data].drop(combination, axis=1))
                        tmp_model.set_params(self.__max_iter, self.__tol)
                        tmp_model_result = tmp_model.run_kmean(initial_setting[train_data][0], initial_setting[train_data][1])
                        
                        cluster = pd.DataFrame(tmp_model_result[2].predict(self.__raw_data[each_data]))
                        label = pd.DataFrame(labels[train_data])

                        cluster_with_label = pd.concat([cluster, label], axis=1)
                        cluster_with_label.columns = ["cluster", "label"]

                        tmp_accuracy_list = []
                        for k in range(initial_setting[train_data][0]):
                            tmp = cluster_with_label[cluster_with_label["cluster"] == k]
                            tmp_accuracy_list.append(tmp["label"].sum()/len(tmp))

                    accuracy_std_list.append(np.std(tmp_accuracy_list))

                break

                while True:
                    tmp_combinations = list(combinations(self.__raw_data[each_data].columns, combinations_cnt))
                    tmp_combinations = [list(tmp) for tmp in tmp_combinations if len((set(tmp)) - set(currently_droped)) != len(tmp)]
                    isStdChanged = False

                    for combination in tmp_combinations:
                        tmp_model = MyKmeans(self.__raw_data[each_data].drop(combination, axis=1))
                        tmp_model.set_params(self.__max_iter, self.__tol)
                        tmp_model_result = tmp_model.run_kmean(initial_setting[train_data][0], initial_setting[train_data][1])
                        
                        cluster = pd.DataFrame(tmp_model_result[2].predict(self.__raw_data[each_data]))
                        label = pd.DataFrame(labels[train_data])

                        cluster_with_label = pd.concat([cluster, label], axis=1)
                        cluster_with_label.columns = ["cluster", "label"]

                        label_accuracy = []
                        for k in range(initial_setting[train_data][0]):
                            tmp = cluster_with_label[cluster_with_label["cluster"] == k]
                            label_accuracy.append(tmp["label"].sum()/len(tmp))

                        tmp_std = np.std(label_accuracy)
                        if tmp_std > max_accuracy_std:
                            max_label_accuracy = label_accuracy
                            max_accuracy_std = tmp_std
                            max_accuracy_combination = combination
                            isStdChanged = True

                    combinations_cnt += 1
                    currently_droped = max_accuracy_combination

                    if not isStdChanged:
                        break

                result_train_data_accuracy[train_data] = {"label_accuracy":max_label_accuracy, "accuracy_std":max_accuracy_std, "combination":max_accuracy_combination}
            
            self.__init_params = initial_setting
            self.__optimized_result = result_train_data_accuracy
            
        print("Jobs Done")

        #         while True:
        #             tmp_combinations = list(combinations(self.__raw_data[each_data].columns, combinations_cnt))
        #             tmp_combinations = [list(tmp) for tmp in tmp_combinations if len((set(tmp)) - set(currently_droped)) != len(tmp)]
        #             isStdChanged = False

        #             for combination in tmp_combinations:
        #                 tmp_model = MyKmeans(self.__raw_data[each_data].drop(combination, axis=1))
        #                 tmp_model.set_params(self.__max_iter, self.__tol)
        #                 tmp_model_result = tmp_model.run_kmean(initial_setting[train_data][0], initial_setting[train_data][1])
                        
        #                 cluster = pd.DataFrame(tmp_model_result[2].predict(self.__raw_data[each_data]))
        #                 label = pd.DataFrame(labels[train_data])

        #                 cluster_with_label = pd.concat([cluster, label], axis=1)
        #                 cluster_with_label.columns = ["cluster", "label"]

        #                 label_accuracy = []
        #                 for k in range(initial_setting[train_data][0]):
        #                     tmp = cluster_with_label[cluster_with_label["cluster"] == k]
        #                     label_accuracy.append(tmp["label"].sum()/len(tmp))

        #                 tmp_std = np.std(label_accuracy)
        #                 if tmp_std > max_accuracy_std:
        #                     max_label_accuracy = label_accuracy
        #                     max_accuracy_std = tmp_std
        #                     max_accuracy_combination = combination
        #                     isStdChanged = True

        #             combinations_cnt += 1
        #             currently_droped = max_accuracy_combination

        #             if not isStdChanged:
        #                 break

        #         result_train_data_accuracy[train_data] = {"label_accuracy":max_label_accuracy, "accuracy_std":max_accuracy_std, "combination":max_accuracy_combination}
            
        #     self.__init_params = initial_setting
        #     self.__optimized_result = result_train_data_accuracy
            
        # print("Jobs Done")

    @property
    def init_params(self):
        try:
            return self.__init_params
        except:
            raise Exception("No optimization result exists")
    
    @property
    def optimized_result(self):
        try:
            return self.__optimized_result
        except:
            raise Exception("No optimization result exists")

