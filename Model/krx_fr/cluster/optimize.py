
import random
from itertools import combinations

import pandas as pd
import numpy as np

from tqdm import tqdm

from .kmeans import MyKmeans

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
        else:
            self.set_optimize_k()
        
        if not self.__isOptimize_init_point:
            if init_point == None:
                raise ValueError("Please set the init_point")
            elif raw_data.keys() != init_point.keys():
                raise ValueError("keys for raw_data and init_point do not match")
        else:
            self.set_optimize_initp()

        self.__raw_data = raw_data
        self.__number_of_cluster = number_of_cluster
        self.__init_point = init_point
    
    def set_optimize_k(self, max_k=10, max_sample=500):
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

            if self.__isOptimize_number_of_cluster:
                tmp_model.find_optimal_k(self.__optimize_k_max_k, self.__optimize_k_max_sample, "silhouette", off_tqdm=True)
                k = None
            else:
                k = self.__number_of_cluster[each_data]
            if self.__isOptimize_init_point:
                tmp_model.find_optimal_initp(k, self.__optimize_initp_max_sample, self.__optimize_initp_method, off_tqdm=True)
                init_point = None
            else:
                init_point = self.__init_point[each_data]
            
            initial_setting[each_data] = [k, init_point]

        if self.__isOptimize_variable:
            random.seed(random_seed)
            train_data_list = sorted(random.sample(list(self.__raw_data.keys()), int(len(self.__raw_data.keys())*train_data_ratio)))

            result_train_data_accuracy = {}

            max_accuracy_std = 0

            combinations_cnt = 0
            # currently_droped = [tmp[0] for tmp in list(combinations(self.__raw_data[each_data].columns, combinations_cnt))]
            while True:
                isCombinationChanged = False
                if combinations_cnt == 0:
                    combination = []
                    accuracy_std_list = []
                    for train_data in train_data_list:
                        tmp_model = MyKmeans(self.__raw_data[train_data])
                        tmp_model.set_params(self.__max_iter, self.__tol)
                        tmp_model_result = tmp_model.run_kmean(initial_setting[train_data][0], initial_setting[train_data][1])
                        
                        cluster = pd.DataFrame(tmp_model_result["model"].predict(self.__raw_data[train_data].values))
                        label = pd.DataFrame(labels[train_data])
                        cluster.columns = ["cluster"]
                        label.columns = ["label"]
                        cluster.index = self.__raw_data[train_data].index
                        label.index = labels[train_data].index

                        cluster_with_label = cluster.join(label).fillna(0)

                        tmp_accuracy_list = []
                        for k in range(initial_setting[train_data][0]):
                            tmp = cluster_with_label[cluster_with_label["cluster"] == k]
                            # print(tmp)
                            # print(tmp["label"].sum()/len(tmp))
                            tmp_accuracy_list.append(tmp["label"].sum()/len(label))

                        accuracy_std_list.append(np.std(tmp_accuracy_list))

                else:
                    tmp_combinations = list(combinations(self.__raw_data[each_data].columns, combinations_cnt))
                    tmp_combinations = [list(tmp) for tmp in tmp_combinations if len((set(tmp)) - set(currently_droped)) == len(tmp)-len(currently_droped)]

                    accuracy_std_list = []
                    # print(tmp_combinations)
                    for combination in tqdm(tmp_combinations, desc="Optimizing variable(depth={})...".format(combinations_cnt)):
                        for train_data in train_data_list:
                            tmp_model = MyKmeans(self.__raw_data[train_data].drop(combination, axis=1))
                            tmp_model.set_params(self.__max_iter, self.__tol)
                            tmp_model_result = tmp_model.run_kmean(initial_setting[train_data][0], initial_setting[train_data][1])
                            
                            cluster = pd.DataFrame(tmp_model_result["model"].predict(self.__raw_data[train_data].drop(combination, axis=1).values))
                            label = pd.DataFrame(labels[train_data])
                            cluster.columns = ["cluster"]
                            label.columns = ["label"]
                            cluster.index = self.__raw_data[train_data].index
                            label.index = labels[train_data].index

                            cluster_with_label = cluster.join(label).fillna(0)

                            tmp_accuracy_list = []
                            for k in range(initial_setting[train_data][0]):
                                tmp = cluster_with_label[cluster_with_label["cluster"] == k]
                                # print(tmp)
                                # print(tmp["label"].sum()/len(tmp))
                                tmp_accuracy_list.append(tmp["label"].sum()/len(label))

                        accuracy_std_list.append(np.std(tmp_accuracy_list))

                # print(np.mean(accuracy_std_list))
                if max_accuracy_std < np.mean(accuracy_std_list):
                    max_accuracy_std = np.mean(accuracy_std_list)
                    best_combination = combination
                    isCombinationChanged = True

                if not isCombinationChanged:
                    result_train_data_accuracy = {"accuracy_std":max_accuracy_std, "combination":best_combination}
                    self.__init_params = initial_setting
                    self.__optimized_result = result_train_data_accuracy
                    break
                else:
                    combinations_cnt += 1
                    currently_droped = best_combination
                    # print(currently_droped)
            
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
        
    @staticmethod
    def eval(model, data, label):
        """
        Args:
            model: fitted된 kmeans 모델
            data (pd.DataFrame): fit할 때 사용한 data
            label (pd.DataFrame): 해당 시점의 라벨
        """
        cluster = pd.DataFrame(model.predict(data.values))
        cluster.index = data.index
        label.index = label.index

        cluster_with_label = pd.concat([cluster, label], axis=1)
        cluster_with_label.columns = ["cluster", "label"]

        cluster_accuracy = {}
        for cluster in sorted(list(set(cluster.values.flatten()))):
            tmp = cluster_with_label[cluster_with_label["cluster"] == cluster]
            cluster_accuracy[cluster] = tmp["label"].sum()/len(label)
        
        original_accuracy = {}
        for key in cluster_accuracy:
            original_accuracy[key] = cluster_accuracy[key]

        #print("Cluster Accuracy Std: {}".format(np.std(list(cluster_accuracy.values()))))

        #("Each Cluster Accuracy: ")
        # print(cluster_accuracy)

        value_sum = np.sum(list(cluster_accuracy.values()))
        for key in cluster_accuracy.keys():
            cluster_accuracy[key] = cluster_accuracy[key]/value_sum

        #print(cluster_accuracy)
        return cluster_accuracy, original_accuracy

