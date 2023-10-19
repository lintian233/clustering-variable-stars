#!/usr/bin/env python
"""
Example of using the hierarchical classifier to classify (a subset of) the digits data set.

Demonstrated some of the capabilities, e.g using a Pipeline as the base estimator,
defining a non-trivial class hierarchy, etc.

"""
import torch
import numpy as np
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import itertools
import os
from sklearn.model_selection import GridSearchCV
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled
from astrofeatures.astrocluster import Config


def confusion_matrix(preds, labels, conf_matrix):
    # preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def str2int_list(l, classes):
    # str 转 int，一个类别转一个数字
    # l是一个列表
    l_int = []
    for i in range(len(l)):
        for j in range(len(classes)):
            if l[i] == classes[j]:
                l_int.append(j)
    return np.array(l_int)


def plot_confusion_matrix(
    cm, classes, title, epoch, normalize=False, cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    # print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines["left"].set_position(("data", left))
    ax.spines["right"].set_position(("data", right))
    for edge_i in ["top", "bottom", "right", "left"]:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = "{:.2f}".format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(
            j,
            i,
            num,
            verticalalignment="center",
            horizontalalignment="center",
            color="white" if num > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("Predicted label")
    plt.xlabel("True label")
    # if not os.path.exists('./'+os.path.basename(__file__).replace('.py', '_image/')):
    #     os.makedirs('./'+os.path.basename(__file__).replace('.py', '_image/'))
    plt.savefig("./result/svm/Confusion_matrix.png")


def load_umap(path):
    class_labels = np.load(path + "class_labels.npy")
    umap_20d_data = np.load(path + "umap_20d_data.npy")
    true_labels = np.load(path + "true_labels.npy")

    # 生成labels字典
    class_labels_dict = {}
    for i in range(len(class_labels)):
        CEP_OHTHER = ["A", "T2", "T110", "T120"]
        CEP_T1F = ["T1F"]
        CEP_T1M = ["T1M"]
        RRC = ["RRC"]
        RRD = ["RRD"]
        RRE = ["RRE"]
        DSCT = ["M", "S"]
        ECL = ["CV", "ELL", "NC"]
        LPV = ["SRV", "OSARG", "MIRA"]
        RRAB = ["RRAB"]
        DPV = ["DPV"]
        ECL_C = ["C"]
        if class_labels[i] in DSCT:
            class_labels_dict[i] = "DSCT"
        if class_labels[i] in ECL or class_labels[i] in DPV:
            class_labels_dict[i] = "ECL-NC/ELL/CV/DPV"
        if class_labels[i] in LPV:
            class_labels_dict[i] = f"LPV-{class_labels[i]}"
        if class_labels[i] in RRAB:
            class_labels_dict[i] = "RRLYR-RRAB"
        if class_labels[i] in ECL_C:
            class_labels_dict[i] = "ECL-C"
        if class_labels[i] in RRC:
            class_labels_dict[i] = "RRLYR-RRC"
        if class_labels[i] in CEP_OHTHER:
            class_labels_dict[i] = "CEP-OHTHER"
        if class_labels[i] in CEP_T1F:
            class_labels_dict[i] = "CEP-T1F"
        if class_labels[i] in CEP_T1M:
            class_labels_dict[i] = "CEP-T1M"
        if class_labels[i] in RRD:
            class_labels_dict[i] = "RRLYR-RRD"
        if class_labels[i] in RRE:
            class_labels_dict[i] = "RRLYR-RRE"
    classes = list(class_labels_dict.values())
    # 合并相同的值classes:
    classes = list(set(classes))
    # 生成true_labels字符串列表

    true_labels_list = []
    for i in range(len(true_labels)):
        true_labels_list.append(class_labels_dict[true_labels[i]])

    return umap_20d_data, true_labels_list, classes


# Used for seeding random state


def classify_aiastro():
    """
            <ROOT>
           /      \
          A        B
         / \       |  \
        1   7      C   9
                 /   \
                3     8

    """
    class_hierarchy = {
        ROOT: ["LPV", "NonLPV"],
        "LPV": ["LPV-MIRA", "NonLPV-MIRA"],
        "NonLPV-MIRA": ["LPV-OSARG", "LPV-SRV"],
        "NonLPV": ["RRLYR/CEP", "NonRRLYR/CEP"],
        "RRLYR/CEP": ["RRLYR", "CEP"],
        "CEP": ["CEP-T1F", "NonT1F"],
        "NonT1F": ["CEP-T1M", "CEP-OHTHER"],
        "RRLYR": ["RRLYR-RRAB", "RRLYR-RRCDE"],
        "RRLYR-RRCDE": ["RRLYR-RRC", "NonRRLYR-RRC"],
        "NonRRLYR-RRC": ["RRLYR-RRD", "RRLYR-RRE"],
        "NonRRLYR/CEP": ["ECL-C", "NonECL-C"],
        "NonECL-C": ["DSCT", "ECL-NC/ELL/CV/DPV"],
    }

    config = Config.instance().classfiy_config

    RANDOM_STATE = config["random_state"]
    basesvm = svm.SVC(
        kernel=config["kernel"],
        probability=True,
        gamma=config["gamma"],
        C=config["C"],
    )

    param_grid = {
        "C": np.logspace(-3, 3, 10),
        "gamma": np.logspace(-3, 3, 10),
    }

    base_estimator = make_pipeline(
        TruncatedSVD(n_components=config["n_components"]), basesvm
    )
    grid_search = GridSearchCV(
        base_estimator, param_grid=param_grid, cv=config["cv"], n_jobs=config["n_jobs"]
    )

    if config["ifgridsearch"]:
        base_estimator = grid_search

    clf = HierarchicalClassifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
    )

    X, y, classes = load_umap("./result/umap_cluster/")
    # cast the targets to strings so we have consistent typing of labels across hierarchy
    y = np.array(y).astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 数字化y_pred和y_test
    y_pred = str2int_list(y_pred, classes)
    y_test = str2int_list(y_test, classes)

    conf_matrix = torch.zeros(13, 13)
    conf_matrix = confusion_matrix(y_pred, y_test, conf_matrix)

    plot_confusion_matrix(
        conf_matrix.numpy(), classes=classes, title="", epoch=0, normalize=False
    )
    print(classification_report(y_test, y_pred, target_names=classes))


if __name__ == "__main__":
    classify_aiastro()
