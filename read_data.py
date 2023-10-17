import astrofeatures as AF
import concurrent.futures as cf

def get_features(filepath):
    return AF.AstroDataFeatures(filepath).INIT()


def list_dir_of_path(path):
    """
    path: the path of directory
    return: the list of directory
    """
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def read_ocvs_data(datasetpath):
    """
    datasetpath: the path of dataset
    返回一个字典 , key是类别, value 是类别数据的相对地址

    datasets 的目录结构有四层, 第一层是类别, 第二层是子类别,第三层是I,V波段,第四层是数据
    """
    dataset = {}

    class_dir = list_dir_of_path(datasetpath)
    for class_name in class_dir:
      dir_path = os.path.join(datasetpath,class_name)
      sub_class_name = list_dir_of_path(dir_path)      
      for sub_class in sub_class_name:
        subdir_path = os.path.join(dir_path,sub_class)
        I_data_path = os.path.join(subdir_path,"I")
        I_file_list = os.listdir(I_data_path)
        I_file_list_path = [os.path.join(I_data_path, f) for f in I_file_list]
        dataset[sub_class] = I_file_list_path

    return dataset

def read_class_data(path_array, class_num):
    arr = np.zeros((len(path_array), 143), dtype=np.float64)
    with cf.ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(get_features, path_array)

    for i, result in enumerate(results):
        arr[i] = result
    
    class_label = np.array([class_num] * len(path_array))
    return arr, class_label

def save_npy(arr, class_label, class_name):
    np.save(f"./npy_data/{class_name}_data.npy", arr)
    np.save(f"./npy_data/{class_name}_label.npy", class_label)


if __name__ == '__main__':
    dataset = read_ocvs_data(r"datasets/OCVS")
    class_num =[i for i in range(len(dataset.keys()))]
    for i in class_num:
        for class_name, path_arr in dataset.items():
            arr, class_label = read_class_data(path_arr, i)
            print(arr.shape)
            save_npy(arr, class_label, class_name)