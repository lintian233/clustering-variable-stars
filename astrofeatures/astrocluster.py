import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        self.path = os.path.join("./", self.config["data_path"])

        self.data = None
        self.class_name = None
        self.class_len = None
        self.visual_data = None
        self.traning_data = None
        self.labels = None

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

    def INIT(self):
        self.__load_data(self.path)
        self.__generate_labels()
        self.__reduce_to_2d()
        self.__reduce_to_20d()
        self.__hdbscan_cluster()
        self.__calculate_purity()
        self.__save_data("./result/umap_cluster/")

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

    def visualize_origin_umap(self):
        index = self.index
        data = self.visual_data
        class_name = self.class_name

        fig, ax = plt.subplots(figsize=(12, 10), dpi=400)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        for i in range(len(index)):
            color = sns.color_palette("Spectral", as_cmap=True)(i / len(index))
            if i == 0:
                sc = ax.scatter(
                    data[: index[i], 0],
                    data[: index[i], 1],
                    label=class_name[i],
                    s=self.config["node_size"],
                    color=color,
                )
            else:
                ax.scatter(
                    data[index[i - 1] : index[i], 0],
                    data[index[i - 1] : index[i], 1],
                    label=class_name[i],
                    s=self.config["node_size"],
                    color=color,
                )
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        sns.despine(left=True, right=True, top=True, bottom=True)
        plt.savefig("./result/visual/umap_origin.png")
        # plt.show()

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
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
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

            plt.savefig(f"./result/visual/temp/{class_name[i]}.png")
            plt.close(fig)

        for i in range(len(class_name)):
            update(i)

        images = []
        for i in range(len(class_name)):
            images.append(imageio.imread(f"./result/visual/temp/{class_name[i]}.png"))

        imageio.mimsave(f"./result/visual/scatter_all_{mode}.gif", images, fps=1)

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
        fig.patch.set_facecolor("black")
        for i in range(len(class_name)):
            currenti = i
            j = np.mod(i, col)

            row = int(i / col)

            all_data = pd.DataFrame(data, columns=["x", "y"])
            index = np.where(labels == class_name[i])
            if (mode == "compare") or (mode == "origin"):
                index = np.where(labels == i)

            spec_data = pd.DataFrame(data[index], columns=["x", "y"])

            axes[row, j].set_facecolor("black")
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
            axes[row, j].set_title(class_name[i], color="white")
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
        plt.savefig(f"./result/visual/scatter_all_{mode}.png")

    def visualize_cluster_umap(self):
        labels = self.predicted_labels
        data = self.visual_data
        node_size = self.config["node_size"]

        num = np.unique(labels)
        fig, ax = plt.subplots(figsize=(12, 10), dpi=400)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

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
        plt.savefig("./result/visual/umap_cluster.png")
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

    def __calculate_purity(self):
        labels = self.labels
        cluster_labels = self.predicted_labels
        class_name = self.class_name

        num = np.unique(labels)
        c_num = np.unique(cluster_labels)
        purity = []
        C_purity = []
        for i in range(1, len(c_num)):
            # N_w 是属于某个簇的个数
            index = np.where(cluster_labels == c_num[i])
            N_w = len(index[0])
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

        plt.savefig("./result/visual/purity.png", dpi=400, bbox_inches="tight")
