import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import matplotlib.animation as animation
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import umap
import json
import subprocess
import hdbscan
import yaml
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="cluster")
    parser.add_argument(
        "-d", "--dataset", type=str, help="Dataset name", default="None"
    )
    return parser.parse_args()


class Config:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            path = os.path.join("./", "config/config.yaml")
            cls._instance._load_parameters(path)

        return cls._instance

    def _load_parameters(self, path):
        config_file_path = path
        with open(config_file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        self.__dict__.update(config)


class Astrocluster:
    def __init__(self):
        self.config = Config.instance().cluster_config

        args = get_args()

        self.path = os.path.join("./npy_data", args.dataset)
        self.result_path = os.path.join("./result", args.dataset)

        self.data = None
        self.class_name = None
        self.class_len = None
        self.visual_data = None
        self.traning_data = None
        self.labels = None
        self.if_classfiy_semi = self.config["if_classfiy_semi"]
        self.if_visual_semi = self.config["if_visual_semi"]
        self.percent_semi = self.config["percent_semi"]
        self.cluster_data = None

        self.index = None
        self.ifscale = self.config["ifscale"]

        self.reducer_visual = umap.UMAP(
            n_neighbors=self.config["n_neighbors"],
            min_dist=self.config["min_dist"],
            n_components=2,
            tqdm_kwds={"disable": True},
        )

        self.reducer_traning = umap.UMAP(
            n_neighbors=self.config["n_neighbors"],
            min_dist=self.config["min_dist"],
            n_components=self.config["n_components"],
            tqdm_kwds={"disable": True},
        )

        self.hdbscan_cluster = hdbscan.HDBSCAN(
            min_cluster_size=self.config["min_cluster_size"],
            min_samples=self.config["min_samples"],
            cluster_selection_epsilon=self.config["cluster_selection_epsilon"],
            cluster_selection_method=self.config["cluster_selection_method"],
        )

        self.predicted_labels = None
        self.C_class = None
        self.purity = None
        self.cluster_num = None

        self.specatur_cluster = SpectralClustering(
            n_clusters=20,
            random_state=42,
        )
        self.predicted_labels_spectral = None

    def INIT(self):
        self.__load_data(self.path)
        self.__generate_labels()
        self.__reduce_to_2d()
        self.__reduce_to_20d()
        self.__spectural_cluster()
        self.__hdbscan_cluster()
        self.__calculate_purity()

        self.__save_data(os.path.join(self.result_path, "umap_cluster"))

        return self
    
    

    def __save_data(self, path):
        np.save(os.path.join(path, "raw_data.npy"), self.data)
        np.save(os.path.join(path, "true_labels.npy"), self.labels)
        np.save(os.path.join(path, "class_labels"), self.class_name)
        np.save(os.path.join(path, "umap_visual_data.npy"), self.visual_data)
        np.save(os.path.join(path, "umap_20d_data.npy"), self.traning_data)
        np.save(os.path.join(path, "hdbscan_cluster_labels.npy"), self.predicted_labels)
        
    def __load_data(self, path):
        file_name = []
        dataset = []
        class_len = []
        labels = []
        for file in os.listdir(path):
            if file.endswith("_data.npy"):
                file_name.append(file)
        for file_path in [os.path.join(path, file) for file in file_name]:
            data = np.load(file_path)
            data = np.nan_to_num(data, nan=np.nanmedian(data))
            class_len.append(data.shape[0])
            dataset.append(data)
            class_name = [file.split("_")[0] for file in file_name]

        self.data = np.concatenate(dataset)
        if self.ifscale:
            self.data = StandardScaler().fit_transform(self.data)

        self.class_name = class_name
        self.class_len = class_len
        self.index = np.cumsum(self.class_len)

    def __spectural_cluster(self):
        self.predicted_labels = self.specatur_cluster.fit_predict(self.traning_data)

    # def visualize_spectural_cluster(self):
    #     self.__spectural_cluster()
    #     self.scatter_all(mode="cluster")
    #     self.scatter_gif(mode="cluster")

    def visualize_origin_umap(self):
        embedding_visual = self.visual_data
        labels = self.labels
        class_name = self.class_name
        node = self.config["node_size"]

        fig, ax = plt.subplots(1, figsize=(12, 8), dpi=400)
        # fig.patch.set_facecolor("black")
        fig.patch.set_facecolor("white")
        # ax.set_facecolor("black")
        ax.set_facecolor("white")

        plt.scatter(*embedding_visual.T, s=node, c=labels, cmap="Spectral", alpha=1.0)

        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(21) - 0.5)
        cbar.set_ticks(np.arange(20))
        cbar.set_ticklabels(class_name, color="black")
        # 去除边框
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.savefig(
            os.path.join(self.result_path, "visual/umap_origin.png"),
            bbox_inches="tight",
            dpi=400,
        )

    def scatter_gif(self, mode="cluster"):
        data = self.visual_data
        node = self.config["node_size"]

        # mode == default : cluster
        class_name = np.unique(self.predicted_labels)
        labels = self.predicted_labels

        if mode == "origin":
            class_name = self.class_name
            labels = self.labels

        if mode == "compare":
            class_name = self.class_name
            labels = self.predicted_labels

        def update(i):
            fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
            # fig.patch.set_facecolor("black")
            # ax.set_facecolor("black")
            ax.set_facecolor("white")
            ax.patch.set_facecolor("white")

            all_data = pd.DataFrame(data, columns=["x", "y"])
            index = np.where(labels == class_name[i])
            if (mode == "compare") or (mode == "origin"):
                index = np.where(labels == i)

            spec_data = pd.DataFrame(data[index], columns=["x", "y"])
            sns.scatterplot(
                x="x", y="y", data=all_data, s=node, color="grey", legend=False, ax=ax
            )

            sns.scatterplot(
                x="x", y="y", data=spec_data, s=node, c="cyan", legend=False, ax=ax
            )

            sns.despine(left=True, right=True, top=True, bottom=True)
            ax.set_title(class_name[i], fontsize=20, color="white")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            path = os.path.join(self.result_path, "visual/temp")
            plt.savefig(f"{path}/{class_name[i]}.png", bbox_inches="tight", dpi=200)
            plt.close(fig)

        for i in range(len(class_name)):
            update(i)

        images = []
        for i in range(len(class_name)):
            path = os.path.join(self.result_path, "visual/temp")
            images.append(imageio.imread(f"{path}/{class_name[i]}.png"))

        path = os.path.join(self.result_path, "visual")
        imageio.mimsave(f"{path}/scatter_{mode}.gif", images, fps=1)

    def scatter_all(self, mode="cluster"):
        data = self.visual_data
        node = self.config["node_size"]

        # mode == default : cluster
        class_name = np.unique(self.predicted_labels)
        labels = self.predicted_labels

        if mode == "origin":
            class_name = self.class_name
            labels = self.labels

        if mode == "compare":
            class_name = self.class_name
            labels = self.predicted_labels

        if mode == "spectural":
            class_name = np.unique(self.predicted_labels_spectral)
            labels = self.predicted_labels_spectral

        # sns.set_style("dark")
        length = len(class_name)
        _row = 4
        r = np.mod(length, _row)
        col = int(length / _row)
        if r != 0:
            col += 1
        currenti = 0
        print(f"row : {_row}, col : {col}")
        fig, axes = plt.subplots(_row, col, figsize=(15, 10), dpi=400)
        # fig.patch.set_facecolor("black")
        fig.patch.set_facecolor("white")
        for i in range(len(class_name)):
            currenti = i
            j = np.mod(i, col)

            row = int(i / col)

            all_data = pd.DataFrame(data, columns=["x", "y"])
            index = np.where(labels == class_name[i])
            if (mode == "compare") or (mode == "origin"):
                index = np.where(labels == i)

            spec_data = pd.DataFrame(data[index], columns=["x", "y"])

            # axes[row, j].set_facecolor("black")
            sns.scatterplot(
                x="x",
                y="y",
                data=all_data,
                s=node,
                color="grey",
                ax=axes[row, j],
                legend=False,
            )

            sns.scatterplot(
                x="x",
                y="y",
                data=spec_data,
                s=node,
                ax=axes[row, j],
                c="cyan",
                legend=False,
            )

            sns.despine(left=True, right=True, top=True, bottom=True)
            axes[row, j].set_title(class_name[i], color="black")
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])
            axes[row, j].set_xlabel("")
            axes[row, j].set_ylabel("")

        if currenti < (_row * col - 1):
            for i in range(currenti + 1, (_row * col)):
                j = np.mod(i, col)
                row = int(i / col)
                axes[row, j].set_visible(False)

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        path = os.path.join(self.result_path, "visual")
        plt.savefig(f"{path}/scatter_{mode}.png", bbox_inches="tight", dpi=400)

    def visualize_cluster_umap(self):
        labels = self.predicted_labels
        data = self.visual_data
        node_size = self.config["node_size"]

        num = np.unique(labels)
        fig, ax = plt.subplots(figsize=(12, 10), dpi=400)
        # fig.patch.set_facecolor("black")
        # ax.set_facecolor("black")

        squares = []
        for i in range(len(num)):
            index = np.where(labels == num[i])
            colors = sns.color_palette("Spectral", as_cmap=True)((i) / len(num))
            ax.scatter(
                data[index, 0], data[index, 1], s=node_size, label=num[i], color=colors
            )

        ax.legend(handles=squares, labels=num)
        sns.despine(left=True, right=True, top=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")

        path = os.path.join(self.result_path, "visual")
        plt.savefig(f"{path}/umap_cluster.png", bbox_inches="tight", dpi=400)
        # plt.show()

    def __reduce_to_2d(self):
        """
        如果if_visual_semi为True，
        那么采用半监督学习的方式进行数据的降维和可视化处理。
        这里通过随机选择一定比例(percent)的数据作为标记数据，其余作为未标记数据。
        使用UMAP的fit_transform方法对标记数据进行降维，并将结果存储在visual_data变量中。
        然后，对未标记数据使用transform方法进行降维。
        如果if_use_target_for_cluster为True，那么使用所有数据和其标签(target)进行UMAP降维，
        并将结果存储在visual_data中。
        否则，仅使用数据进行无监督的UMAP降维。
        """
        labels = self.labels
        target = labels
        percent = self.percent_semi
        data = self.data
        if_use_target_for_cluster = self.config["if_use_target_for_cluster"]
        # UMAP将-1标签解释为未标记的点，并相应地进行学习
        
        if self.if_visual_semi:
            length = len(labels)
            visual_data = np.zeros((length, 2))
            index = np.random.choice(length, size=int(percent * length), replace=False)
            # 除去data index 对应的 index
            labeled_index = np.delete(np.arange(length), index)
            visual_data[labeled_index] = self.reducer_visual.fit_transform(
                data[labeled_index], y=target[labeled_index]
            )
            visual_data[index] = self.reducer_visual.transform(data[index])
            self.visual_data = visual_data
        elif if_use_target_for_cluster:
            self.visual_data = self.reducer_visual.fit_transform(data, y=target)
        else:
            self.visual_data = self.reducer_visual.fit_transform(data)

    def __reduce_to_20d(self):
        """
        当if_classfiy_semi为True时，方法执行半监督学习的过程。
        首先，根据percent确定哪些数据点被视为有标签数据，
        然后对这部分数据进行UMAP降维（fit_transform），而剩余的数据点则通过transform进行降维，
        这样可以在保留数据全局结构的同时，对有标签的数据进行特别的处理。
        如果if_classfiy_semi为False，那么对所有数据点进行统一的降维处理，不区分有标签和无标签数据。
        if_use_target_for_cluster为True时，降维过程会考虑target进行聚类优化，
        否则仅根据数据本身特性进行无监督的降维。

        """
        # UMAP将-1标签解释为未标记的点，并相应地进行学习
        labels = self.labels
        data = self.data
        target = labels
        percent = self.percent_semi
        if_use_target_for_cluster = self.config["if_use_target_for_cluster"]

        # if_classfiy_semi 为True时，使用半监督学习
        if self.if_classfiy_semi:
            #
            length = len(labels)
            traning_data = np.zeros((length, self.config["n_components"]))
            index = np.random.choice(length, size=int(percent * length), replace=False)
            labeled_index = np.delete(np.arange(length), index)
            traning_data[labeled_index] = self.reducer_traning.fit_transform(
                data[labeled_index], y=target[labeled_index]
            )
            traning_data[index] = self.reducer_traning.transform(data[index])
            self.traning_data = traning_data
        else:
            #监督学习
            self.traning_data = self.reducer_traning.fit_transform(data, y=target)

        # if_use_target_for_cluster 为True时，使用target进行聚类
        if if_use_target_for_cluster:
            self.cluster_data = self.reducer_traning.fit_transform(data, y=target)
        else:
            self.cluster_data = self.reducer_traning.fit_transform(data)

    def __generate_labels(self):
        labels = []
        for i in range(len(self.class_len)):
            labels += [i] * self.class_len[i]
        self.labels = np.array(labels)

    def __hdbscan_cluster(self):
        self.predicted_labels = self.hdbscan_cluster.fit_predict(self.cluster_data)

    def __calculate_purity(self):
        labels = self.labels
        cluster_labels = self.predicted_labels
        class_name = self.class_name

        num = np.unique(labels)
        c_num = np.unique(cluster_labels)
        cluster_num = []
        purity = []
        C_purity = []
        for i in range(1, len(c_num)):
            # N_w 是属于某个簇的个数
            index = np.where(cluster_labels == c_num[i])
            N_w = len(index[0])
            cluster_num.append(N_w)
            C = []
            for j in range(1, len(num)):
                # index 是cluster_labels中属于某个簇，比如簇0的索引
                index = np.where(cluster_labels == c_num[i])
                # cluster_labels_sub 是属于某个簇的真实标签
                cluster_labels_sub = labels[index]
                # C_i 是属于某个簇的真实标签中，属于某个类别的个数
                C_i = len(np.where(cluster_labels_sub == num[j])[0])
                C.append(C_i)
            C_max = np.max(C)
            # 同时给出C_max所对应的类别
            # print(C)
            C_max_index = np.argmax(C)
            # print(C_max_index)
            C_max_class = class_name[C_max_index]
            # print(C_max_class)
            C_purity.append(C_max_class)
            P = (1 / N_w) * C_max
            purity.append(P)
        self.cluster_num = cluster_num
        self.C_class = C_purity
        self.purity = purity

    def plot_purity(self):
        sns.set_theme(style="ticks")

        # Initialize the figure with a logarithmic x axis
        f, ax = plt.subplots(figsize=(5, 10), dpi=400)
        # ax.set_xscale("log")
        ax.set_xlim(0, 1)

        purity = np.array(self.purity)
        C_purity = np.array(self.C_class)

        df = pd.DataFrame({"Purity": purity, "Class": C_purity})
        sns.boxplot(
            x="Purity",
            y="Class",
            data=df,
            whis=[0, 100],
            width=0.3,
            color="#e0ffff",
            ax=ax,
        )

        path = os.path.join(self.result_path, "visual")
        plt.savefig(f"{path}/purity.png", bbox_inches="tight", dpi=400)
