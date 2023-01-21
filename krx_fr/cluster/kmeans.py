
import os

import pandas as pd
import numpy as np

from numba import njit
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = '10'
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import chart_studio
import plotly.express as px

@njit(cache=True)
def cal_dist(center, pointList):
    """다른 함수에서 참고하는 거리 계산용
    """
    result = []
    for point in pointList:
        result.append(np.linalg.norm(center - point))
    return result

class MyKmeans:

    def __init__(self, data):
        """데이터를 넣어주며 initiate
        
        Args:
            data (pd.DataFrame): (index가 종목명, column이 변수인 데이터프레임)

        """
        self.__data = data.values
        self.__index = data.index
        self.__columns = data.columns

        self.__optimal_random_state = None

        self.set_params()

    def set_params(
        self,
        max_iter = 500,
        tol = 1e-4,
    ):
        """파라미터 설정
        
        Args:
            max_iter (int): 알고리즘의 최대 반복 횟수
            tol (float): loss가 tol 이하일 경우 알고리즘 반복 중단
        """
        self.__max_iter = max_iter
        self.__tol = tol

    def find_optimal_k(
        self,
        max_k,
        max_sample,
        optimize_method = "inertia"
    ):
        """Optimal Number of Cluster를 찾아준다.
        
        Args:
            max_k (int): 1부터 max_k개의 클러스터들을 분석함
            max_sample (int): max_sample번의 시뮬레이션들 돌림
            optimize_method ("intertia", "inter_std","silhouette"): 어떤 방식으로 k를 결정할지
        
        Return:
            dictionary (optimize_method="inertia"): key는 key번째 시뮬레이션, value는 length가 max_k인 리스트
            dictionary (optimize_method="inter_std"): key는 ("dist_mean","dist_std"), value는 해당 값을 지니는 length가 max_k인 리스트

        """
        if optimize_method == "inertia":
            result = {}
            for trial in tqdm(range(max_sample)):
                each_trial = []
                for k in range(2, max_k+1):
                    model = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=self.__max_iter, tol=self.__tol, random_state=trial)
                    model.fit(self.__data)
                    each_trial.append(model.inertia_)
                result[trial] = each_trial

        elif optimize_method == "silhouette":
            result = {}
            for trial in tqdm(range(max_sample)):
                each_trial = []
                for k in range(2, max_k+1):
                    if k == 1:
                        each_trial.append(0)
                    else:
                        model = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=self.__max_iter, tol=self.__tol, random_state=trial)
                        model.fit(self.__data)
                        each_trial.append(silhouette_score(self.__data, model.labels_))
                result[trial] = each_trial
            
            tmp = []
            for j in range(len(result[0])):
                tmp_tmp = []
                for i in range(len(result)):
                    tmp_tmp.append(result[i][j])
                tmp.append(tmp_tmp)

            tmp = [np.nanmean(a) for a in tmp]
            self.__optimal_k = tmp.index(np.max(tmp))+2
                
        elif optimize_method == "inter_std":
            result = {"dist_mean":[], "dist_std":[]}
            for trial in tqdm(range(max_sample)):
                distance_mean = []
                distance_std = []
                for k in range(2, max_k+1):
                    model = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=self.__max_iter, tol=self.__tol, random_state=trial)
                    model.fit(self.__data)

                    cluster_result = model.predict(self.__data) # 각점의 클러스터 라벨
                    number_index = [ttt for ttt in range(cluster_result.shape[0])]
                    cluster_df = pd.DataFrame(cluster_result, index=number_index, columns=['cluster'])
                    tmp_df = pd.DataFrame(self.__data, index=number_index)
                    cluster_df = pd.concat([tmp_df, cluster_df], axis=1)

                    centroid = model.cluster_centers_
                    centroid_result = model.predict(centroid) # 중심점의 클러스터 라벨
                    number_index = [ttt for ttt in range(centroid_result.shape[0])]
                    centroid_df = pd.DataFrame(centroid, index=number_index)
                    tmp_df = pd.DataFrame(centroid_result, index=number_index, columns=['cluster'])
                    centroid_df = pd.concat([centroid_df, tmp_df], axis=1)

                    distance_list = []
                    for j in range(centroid_df.shape[0]):
                        target_cluster = centroid_df.iloc[j,-1]
                        centroid_point = np.array(centroid_df.iloc[j,0:len(centroid_df.columns)-1])
                        each_points = np.array(cluster_df[cluster_df['cluster']==target_cluster].iloc[:,0:len(cluster_df.columns)-1])

                        each_iter = cal_dist(centroid_point, each_points) # 클러스터 중심점과 각 점마다가의 거리를 계산하여, 리스트로 표현, 원소 하나가 거리 하나
                        distance_list.append(each_iter) # jth 클러스터 안의 거리들을 하나의 리스트에 넣어줌

                    distance_list = [item for subList in distance_list for item in subList] # (리스트 안의) 리스트들을 풀어서 전부 각각의 원소로 바꾸어줌
                    distance_mean.append(np.mean(distance_list))
                    distance_std.append(np.std(distance_list))
                
                result["dist_mean"].append(distance_mean) # distance_mean이라는 length가 max_k인 리스트를 원소로 가지는 리스트 (length가 max_sample)
                result["dist_std"].append(distance_std)
        else:
            pass
        
        return result
    
    def find_optimal_initp(
        self,
        num_of_cluster,
        max_sample,
        optimize_method = "inertia"
    ):
        """K-means 클러스터링을 위한 최적의 시작점을 찾는다

        Args:
            num_of_cluster (int): 클러스터 개수
            max_sample (int): 반복할 시뮬레이션 횟수
            optimize_method ("intertia", "inter_std","silhouette"): 어떤 방식으로 시작점을 결정할지

        Return:
            dictionary (optimize_method="inertia"): key는 key번째 시뮬레이션, value는 length가 max_k인 리스트
            dictionary (optimize_method="inter_std"): key는 ("dist_mean","dist_std"), value는 해당 값을 지니는 length가 max_k인 리스트

        """
        if optimize_method == "inertia":
            result = {}
            for trial in tqdm(range(max_sample)):
                model = KMeans(n_clusters=num_of_cluster, init="k-means++", n_init=1, max_iter=self.__max_iter, tol=self.__tol, random_state=trial)
                model.fit(self.__data)
                result[trial] = model.inertia_
            self.__optimal_random_state = [trial for trial in result.keys() if result[trial] == np.min(list(result.values()))][0]

        elif optimize_method == "silhouette":
            result = {}
            for trial in tqdm(range(max_sample)):
                model = KMeans(n_clusters=num_of_cluster, init="k-means++", n_init=1, max_iter=self.__max_iter, tol=self.__tol, random_state=trial)
                model.fit(self.__data)
                result[trial] = silhouette_score(self.__data, model.labels_)
            self.__optimal_random_state = [trial for trial in result.keys() if result[trial] == np.min(list(result.values()))][0]

        elif optimize_method == "inter_std":
            result = {"dist_mean":[], "dist_std":[]}
            for trial in tqdm(range(max_sample)):
                distance_mean = []
                distance_std = []

                model = KMeans(n_clusters=num_of_cluster, init="k-means++", n_init=1, max_iter=self.__max_iter, tol=self.__tol, random_state=trial)
                model.fit(self.__data)

                cluster_result = model.predict(self.__data) # 각점의 클러스터 라벨
                number_index = [ttt for ttt in range(cluster_result.shape[0])]
                cluster_df = pd.DataFrame(cluster_result, index=number_index, columns=['cluster'])
                tmp_df = pd.DataFrame(self.__data, index=number_index)
                cluster_df = pd.concat([tmp_df, cluster_df], axis=1)

                centroid = model.cluster_centers_
                centroid_result = model.predict(centroid) # 중심점의 클러스터 라벨
                number_index = [ttt for ttt in range(centroid_result.shape[0])]
                centroid_df = pd.DataFrame(centroid, index=number_index)
                tmp_df = pd.DataFrame(centroid_result, index=number_index, columns=['cluster'])
                centroid_df = pd.concat([centroid_df, tmp_df], axis=1)

                distance_list = []
                for j in range(centroid_df.shape[0]):
                    target_cluster = centroid_df.iloc[j,-1]
                    centroid_point = np.array(centroid_df.iloc[j,0:len(centroid_df.columns)-1])
                    each_points = np.array(cluster_df[cluster_df['cluster']==target_cluster].iloc[:,0:len(cluster_df.columns)-1])

                    each_iter = cal_dist(centroid_point, each_points) # 클러스터 중심점과 각 점마다가의 거리를 계산하여, 리스트로 표현, 원소 하나가 거리 하나
                    distance_list.append(each_iter) # jth 클러스터 안의 거리들을 하나의 리스트에 넣어줌

                distance_list = [item for subList in distance_list for item in subList] # (리스트 안의) 리스트들을 풀어서 전부 각각의 원소로 바꾸어줌
                distance_mean.append(np.mean(distance_list))
                distance_std.append(np.std(distance_list))
            
                result["dist_mean"].append([trial, distance_mean]) # distance_mean이라는 length가 max_k인 리스트를 원소로 가지는 리스트 (length가 max_sample)
                result["dist_std"].append([trial, distance_std])

            minium_std = np.min([dist[1][0] for dist in result["dist_std"]])
            self.__optimal_random_state = [simul[0] for simul in result["dist_std"] if simul[1][0] == minium_std][0]

        return result
        
    def run_kmean(
        self,
        num_of_cluster=None,
        random_state=None
    ):
        """K-Mean 클러스터링을 돌린다

        Args:
            num_of_cluster (int): 클러스터 개수
            random_state (int): 클러스터링을 시작하는 random initial point .find_optimal_initp로 최적화 가능

        Return:
            주어진 데이터로 fitted 된 k-means model
        """
        if num_of_cluster == None:
            num_of_cluster = self.__optimal_k
        if random_state == None:
            random_state = self.__optimal_random_state

        model = KMeans(n_clusters=num_of_cluster, init="k-means++", n_init=1, max_iter=self.__max_iter, tol=self.__tol, random_state=random_state)
        model.fit(self.__data)

        result = {
            "num_of_cluseer":num_of_cluster,
            "random_state":random_state,
            "model":model
        }
        return result
    
    @staticmethod
    def distance_decomposition(
        fitted_model
    ):
        """클러스터 중심점 간의 거리를 각 변수(차원, 축)별로 분해한다

        Args:
            fitted_model (): .run_kmeans에서 return된 모델

        Return:
            삼중리스트 [[[]]], li[i][j]를 통해 클러스터 i와 j의 거리가 분해된 것을 확인 가능, 대각행렬임
        """
        result = {}
        centroids = fitted_model.cluster_centers_
        for id, centroid in enumerate(centroids):
            result["centroid_{}".format(id)] = centroid

        distance_matrix = np.zeros((centroids.shape[0], centroids.shape[0]))
        for i, centroid_0 in enumerate(centroids):
            for j, centorid_1 in enumerate(centroids):
                distance_matrix[i,j] = np.linalg.norm(centroid_0 - centorid_1)
        distance_matrix = distance_matrix*distance_matrix

        axis_delta_matrix = [[[] for i in range(centroids.shape[0])] for j in range(centroids.shape[0])]
        for i, centroid_0 in enumerate(centroids):
            for j, centorid_1 in enumerate(centroids):
                tmp_list = []
                for axis in range(centroids.shape[1]):
                    each_delta = np.linalg.norm(centroid_0[axis] - centorid_1[axis])
                    tmp_list.append(each_delta*each_delta)
                axis_delta_matrix[i][j] = tmp_list
        
        decomposed_distance_matrix = [[[] for i in range(centroids.shape[0])] for j in range(centroids.shape[0])]
        for i, centroid_0 in enumerate(centroids):
            for j, centorid_1 in enumerate(centroids):
                if i == j:
                    continue
                decomposed_distance_matrix[i][j] = list(np.array(axis_delta_matrix[i][j])/distance_matrix[i,j])

        return decomposed_distance_matrix

    @staticmethod
    def visualize_2d(
        raw_data,
        fitted_model,
        demensional_reduction_model = "pca",
        fig_title = "figure",
        online = False,
    ):
        if type(raw_data) == pd.core.frame.DataFrame:
            raw_data = raw_data.values

        if demensional_reduction_model == "mds":
            demensional_reduction_model = MDS(n_components=2)
            tmp_columns = ["MD1", "MD2"]
        elif demensional_reduction_model == "pca":
            demensional_reduction_model = PCA(n_components=2)
            tmp_columns = ["PC1", "PC2"]
        
        labels = pd.DataFrame(fitted_model.predict(raw_data), columns=["label"])
        reducted_points = pd.DataFrame(demensional_reduction_model.fit_transform(raw_data), columns=tmp_columns)
        
        result_df = pd.concat([reducted_points, labels], axis=1)

        fig = px.scatter(result_df, x=tmp_columns[0], y=tmp_columns[1], color='label', title=fig_title)
        
        if online:
            chart_studio.plotly.plot(fig, auto_open=True)
        else:
            fig.write_html("fig.html", auto_open=True)

    @staticmethod
    def visualize_3d(
        raw_data,
        fitted_model,
        demensional_reduction_model = "pca",
        fig_title = "figure",
        online = False,
    ):
        if type(raw_data) == pd.core.frame.DataFrame:
            raw_data = raw_data.values

        if demensional_reduction_model == "mds":
            demensional_reduction_model = MDS(n_components=3)
            tmp_columns = ["MD1", "MD2", "MD3"]
        elif demensional_reduction_model == "pca":
            demensional_reduction_model = PCA(n_components=3)
            tmp_columns = ["PC1", "PC2", "PC3"]
        
        labels = pd.DataFrame(fitted_model.predict(raw_data), columns=["label"])
        reducted_points = pd.DataFrame(demensional_reduction_model.fit_transform(raw_data), columns=tmp_columns)
        
        result_df = pd.concat([reducted_points, labels], axis=1)

        fig = px.scatter_3d(result_df, x=tmp_columns[0], y=tmp_columns[1], z=tmp_columns[2], color='label', title=fig_title)
        
        if online:
            chart_studio.plotly.plot(fig, auto_open=True)
        else:
            fig.write_html("fig.html", auto_open=True)