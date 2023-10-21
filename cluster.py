import astrofeatures.astrocluster as ac
import numpy as np
import pandas as pd

def purity_table(purity, C_purity):
    purity = np.array(purity)
    C_purity = np.array(C_purity)
    class_name = np.unique(C_purity)
    #print(class_name)
    purity_table = []
    for i in range(len(class_name)):
        index = np.where(C_purity == class_name[i])
        purity_table.append(np.mean(purity[index]))
    
    df = pd.DataFrame(purity_table, index=class_name)
    df.columns = ["Purity"]
    df.index.name = "Class"
    return df

def cluster():
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
    table = purity_table(purity, C_purity)
    pd.DataFrame.to_csv(table, "./result/purity_table.csv")

def transform_percent(path):
    df = pd.read_csv(path)
    
    purity = df['Purity'].apply(lambda x: format(x, '.2%'))
    class_ = df['Class']
    df = pd.DataFrame({'Class': class_, 'Purity': purity})
    df.to_csv(path, index=False)



if __name__ == "__main__":
    #cluster()
    transform_percent("./result/purity_table.csv")