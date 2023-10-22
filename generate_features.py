import astrofeatures as AF
import concurrent.futures as cf
import numpy as np
import os
import argparse as ap


def get_args():
    parser = ap.ArgumentParser(description="Read data from dataset")
    parser.add_argument("-c", "--class_name", type=str, help="Class name")
    parser.add_argument(
        "-d", "--dataset", type=str, help="Dataset name", default="None"
    )
    return parser.parse_args()


def get_features(filepath):
    return AF.AstroDataFeatures(filepath).INIT()


def list_dir_of_path(path):
    """
    path: the path of directory
    return: the list of directory
    """
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def read_dataset_data(datasetpath):
    """
    datasetpath: the path of dataset
    返回一个字典 , key是类别, value 是类别数据的相对地址

    datasets 的目录结构有四层, 第一层是类别, 第二层是子类别,第三层是I,V波段,第四层是数据
    """
    dataset = {}

    class_dir = list_dir_of_path(datasetpath)
    for class_name in class_dir:
        dir_path = os.path.join(datasetpath, class_name)
        sub_class_name = list_dir_of_path(dir_path)
        for sub_class in sub_class_name:
            subdir_path = os.path.join(dir_path, sub_class)
            I_data_path = os.path.join(subdir_path, "I")
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


def save_npy(arr, class_label, class_name, path):
    np.save(f"{path}/{class_name}_data.npy", arr)
    np.save(f"{path}/{class_name}_labels.npy", class_label)


def Read(class_name: str):
    args = get_args()
    dataset_name = args.dataset
    dataset_path = os.path.join(os.getcwd(), "datasets", dataset_name)
    print(f"Reading {class_name} data...")
    dataset = read_dataset_data(dataset_path)

    class_num = [i for i in dataset.keys()].index(class_name)
    data_arr_path = dataset[class_name]

    print(f"{class_name} data num: {len(data_arr_path)}")

    data_arr, class_label = read_class_data(data_arr_path, class_num)

    print(f"{class_name} read done!")
    print(f"{class_name} data shape: {data_arr.shape}")

    print(f"Saving {class_name} data...")

    save_npy(data_arr, class_label, class_name)
    print(f"{class_name} save done!")


if __name__ == "__main__":
    args = get_args()
    class_name = args.class_name
    Read(class_name)
