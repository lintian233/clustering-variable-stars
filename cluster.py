import astrofeatures.astrocluster as ac


def cluster():
    a = ac.Astrocluster().INIT()
    a.visualize_origin_umap()
    a.scatter_all(mode="cluster")
    a.scatter_all(mode="origin")
    a.visualize_cluster_umap()


if __name__ == "__main__":
    cluster()