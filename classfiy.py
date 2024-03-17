#!/usr/bin/env python
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
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="classify")
    parser.add_argument(
        "-d", "--dataset", type=str, help="Dataset name", default="None"
    )

    return parser.parse_args()


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

def plot_confusion_matrix(cm, classes, title, epoch, dataset_name, normalize=False, cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))  # Adjust figure size here
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentage
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # Dynamically adjust font size based on the number of classes
    if len(classes) < 10:
        fontsize = "medium"
    elif len(classes) < 20:
        fontsize = "small"
    else:
        fontsize = "x-small"

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            num = '{:.2f}%'.format(cm[i, j])  # Rounded to one decimal place
        else:
            num = '{}'.format(int(cm[i, j]))
        
        plt.text(j, i, num, verticalalignment="center", horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Create directory if it does not exist
    path = f"./result/{dataset_name}/svm/"
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the figure
    filename = f"confusion_matrix.png"
    plt.savefig(os.path.join(path, filename), dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory


def load_umap(path):
    class_labels = np.load(path + "class_labels.npy")
    umap_20d_data = np.load(path + "umap_20d_data.npy")
    true_labels = np.load(path + "true_labels.npy")

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
    classes = list(set(classes))

    true_labels_list = []
    for i in range(len(true_labels)):
        true_labels_list.append(class_labels_dict[true_labels[i]])

    return umap_20d_data, true_labels_list, classes

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
    args = get_args()
    path = f"./result/{args.dataset}/umap_cluster/"

    X, y, classes = load_umap(path)
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
        conf_matrix.numpy(),
        classes=classes,
        title="",
        epoch=0,
        normalize=True,
        dataset_name=args.dataset,
    )
    print(classification_report(y_test, y_pred, target_names=classes))


if __name__ == "__main__":
    classify_aiastro()
