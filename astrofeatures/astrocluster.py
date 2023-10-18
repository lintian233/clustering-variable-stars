import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import umap
import json
import subprocess
import hdbscan


class Config:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_parameters()
            cls._instance._openai_init()

        return cls._instance

    def _load_parameters(self):
        config_file_path = r"../config/cluster_config.json"
        with open(config_file_path, "r") as f:
            config = json.load(f)
        self.__dict__.update(config)

    def _openai_init(self):
        config = Config.instance()


class Astrocluster:
    def __init__(self):
        self.config = Config.instance()
        self.path = os.path.join("../", self.config.data_path)

        self.data = None
        self.class_name = None
        self.class_len = None
        self.visual_data = None
        self.traning_data = None
        self.labels = None

        self.index = None
        self.ifscale = self.config.ifscale

        self.reducer_visual = umap.UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            n_components=2,
            tqdm_kwds={"disable": True},
        )

        self.reducer_traning = umap.UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            n_components=self.config.n_components,
            tqdm_kwds={"disable": True},
        )

        self.hdbscan_cluster = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            cluster_selection_method=self.config.cluster_selection_method,
        )

        self.predicted_labels = None

    def INIT(self):
        self.__load_data(self.path)
        self.__generate_labels()
        self.__reduce_to_2d()
        self.__reduce_to_20d()
        self.__hdbscan_cluster()

        return self

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

    def visualize_origin_umap(self):
        index = self.index
        data = self.visual_data
        class_name = self.class_name

        fig, ax = plt.subplots(figsize=(12, 10), dpi=400)

        for i in range(len(index)):
            color = sns.color_palette("Spectral", as_cmap=True)(i / len(index))
            if i == 0:
                sc = ax.scatter(
                    data[: index[i], 0],
                    data[: index[i], 1],
                    label=class_name[i],
                    s=self.config.node_size,
                    color=color,
                )
            else:
                ax.scatter(
                    data[index[i - 1] : index[i], 0],
                    data[index[i - 1] : index[i], 1],
                    label=class_name[i],
                    s=self.config.node_size,
                    color=color,
                )
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        sns.despine(left=True, right=True, top=True, bottom=True)
        plt.savefig("../result/visual/umap_origin.png")
        # plt.show()

    def scatter_all(self, mode="cluster"):
        data = self.visual_data
        node = self.config.node_size

        # mode == default : cluster
        class_name = np.unique(self.predicted_labels)
        labels = self.predicted_labels

        if mode == "origin":
            class_name = self.class_name
            labels = self.labels

        if mode == "compare":
            class_name = self.class_name
            labels = self.predicted_labels

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
        for i in range(len(class_name)):
            currenti = i
            j = np.mod(i, col)

            row = int(i / col)

            all_data = pd.DataFrame(data, columns=["x", "y"])
            index = np.where(labels == class_name[i])
            if (mode == "compare") or (mode == "origin"):
                index = np.where(labels == i)

            spec_data = pd.DataFrame(data[index], columns=["x", "y"])
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
            axes[row, j].set_title(class_name[i])
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
        plt.savefig(f"../result/visual/scatter_all_{mode}.png")

    def visualize_cluster_umap(self):
        labels = self.predicted_labels
        data = self.visual_data
        node_size = self.config.node_size

        num = np.unique(labels)
        fig, ax = plt.subplots(figsize=(12, 10), dpi=400)
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
        plt.savefig("../result/visual/umap_cluster.png")
        # plt.show()

    def __reduce_to_2d(self):
        self.visual_data = self.reducer_visual.fit_transform(self.data, y=self.labels)

    def __reduce_to_20d(self):
        self.traning_data = self.reducer_traning.fit_transform(self.data, y=self.labels)

    def __generate_labels(self):
        labels = []
        for i in range(len(self.class_len)):
            labels += [i] * self.class_len[i]
        self.labels = np.array(labels)

    def __hdbscan_cluster(self):
        self.predicted_labels = self.hdbscan_cluster.fit_predict(self.traning_data)
