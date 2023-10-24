import astrofeatures.astrocluster as ac
import numpy as np
import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="cluster")
    parser.add_argument(
        "-d", "--dataset", type=str, help="Dataset name", default="None"
    )

    return parser.parse_args()


def purity_table(purity, C_purity, cluster_num):
    purity = np.array(purity)
    C_purity = np.array(C_purity)
    cluster_num = np.array(cluster_num)
    class_name = np.unique(C_purity)
    # print(class_name)
    purity_table = []
    for i in range(len(class_name)):
        index = np.where(C_purity == class_name[i])
        sepc_num = cluster_num[index]
        purity_num = purity[index]
        purity_table.append(np.sum(sepc_num * purity_num) / np.sum(sepc_num))

    df = pd.DataFrame(purity_table, index=class_name)
    df.columns = ["Purity"]
    df.index.name = "Class"
    df["Purity"] = df["Purity"].apply(lambda x: format(x, ".2%"))
    return df


def cluster():
    args = get_args()
    a = ac.Astrocluster().INIT()

    a.visualize_origin_umap()
    a.scatter_all(mode="cluster")
    a.scatter_all(mode="origin")
    a.visualize_cluster_umap()

    a.scatter_gif(mode="cluster")
    a.scatter_gif(mode="origin")
    a.plot_purity()

    purity = a.purity
    C_purity = a.C_class
    cluster_num = a.cluster_num
    table = purity_table(purity, C_purity, cluster_num)
    path = f"./result/{args.dataset}/purity_table.csv"

    pd.DataFrame.to_csv(table, path, encoding="utf-8-sig")


if __name__ == "__main__":
    cluster()
