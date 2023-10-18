import astrocluster as ac



def test():
    a = ac.Astrocluster().INIT()
    a.visualize_origin_umap()
    a.scatter_all()
    a.visualize_cluster_umap()




if __name__ == "__main__":
    test()