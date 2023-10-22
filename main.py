#!/usr/bin/env python
import subprocess as sp
import os
import errno
import sys
import argparse
import numpy as np

def print_red(text, *args, **kwargs):
    print(f"\033[31m{text}\033[0m", *args, **kwargs)

def print_green(text, *args, **kwargs):
    print(f"\033[32m{text}\033[0m", *args, **kwargs)

def print_bule(text, *args, **kwargs):
    #加粗
    print(f"\033[1;34m{text}\033[0m", *args, **kwargs)


def print_title():
    str = """
                   _            __               
 _   ______ ______(_)     _____/ /_____ ______   
| | / / __ `/ ___/ /_____/ ___/ __/ __ `/ ___/   
| |/ / /_/ / /  / /_____(__  ) /_/ /_/ / /      
|___/\__,_/_/  /_/     /____/\__/\__,_/_/        
                                                 
     """
    print_bule(str)

    str_detail = """
    This is a program for clustering and classifying astronomical variable stars.
    This program is a reproduction of this paper: 
    It is developed by lintian233.
    github:https://github.com/lintian233/clustering-variable-stars
    Please visit the configuration file to modify the hyperparameters when necessary.
    Read the README.md for more information.
    """

    print_bule(str_detail)

def get_args():
    parser = argparse.ArgumentParser(description="cluster and classify")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset name", default="None")

    return parser.parse_args()
def get_class_name(datasetpath):
    class_name_list = []
    #大类别的名称
    class_dir = os.listdir(datasetpath)
    class_dir = [f for f in class_dir if os.path.isdir(os.path.join(datasetpath, f))]
    for class_name in class_dir:
        sub_class_name = os.listdir(os.path.join(datasetpath, class_name))
        sub_class_name = [f for f in sub_class_name if os.path.isdir(os.path.join(datasetpath, class_name, f))]
        for sub_class in sub_class_name:
            class_name_list.append(sub_class)
    return class_name_list

def isexist_feature(dataset):
    npy_data_path = os.path.join(os.getcwd(), "npy_data")
    if not os.path.exists(npy_data_path):
        return False
    subdir = os.listdir(npy_data_path)
    if dataset not in subdir:
        return False
    
    return True

def generate_features(dataset_name_list,dataset_name):
    print_green("Generating features...")
    print_green(f"datalist: {dataset_name_list}")

    #获得threads的数量
    threads = os.cpu_count()
    if threads == 1:
        print_red("damn!, your computer is too old!")
        print_red("the number of threads is 1, please check your computer!")
        exit(errno.ENOENT)

    if threads < 4:
        print_red("warning: the number of threads is less than 4, it will take a long time to generate features!")
        print_green("Do you want to continue? (y/n)")
        choice = input()
        if choice not in ["y", "Y"]:
            exit(0)
     
    l = len(dataset_name_list)
    count = 0
    while count + 4 < l:
        p1 = sp.Popen(f"python generate_features.py -c {dataset_name_list[count]} -d {dataset_name}", shell=True)
        p2 = sp.Popen(f"python generate_features.py -c {dataset_name_list[count+1]} -d {dataset_name}", shell=True)
        p3 = sp.Popen(f"python generate_features.py -c {dataset_name_list[count+2]} -d {dataset_name}", shell=True)
        p4 = sp.Popen(f"python generate_features.py -c {dataset_name_list[count+3]} -d {dataset_name}", shell=True)
        p1.wait()
        p2.wait()
        p3.wait()
        p4.wait()
        if p1.returncode != 0 or p2.returncode != 0 or p3.returncode != 0 or p4.returncode != 0:
            exit(errno.ENOENT)
        count += 4

    #popen
    p_l = []
    while count < l:
        p = sp.Popen(f"python generate_features.py -c {dataset_name_list[count]} -d {dataset_name}", shell=True)
        p_l.append(p)
        count += 1
    
    for p in p_l:
        p.wait()
        if p.returncode != 0:
            exit(errno.ENOENT)

def check_dir_exist():
    current_dir = os.getcwd()
    subdir = os.listdir(current_dir)
    subdir = [f for f in subdir if os.path.isdir(os.path.join(current_dir, f))]

    #检查是否有datasets文件夹
    if "datasets" not in subdir:
        print_red("error: dataset not found!", file=sys.stderr)
        exit(errno.ENOENT)
    
    #检查config文件夹
    if "config" not in subdir:
        print_red("error: config not found!", file=sys.stderr)
        exit(errno.ENOENT)

    #检查是否有指定文件夹
    args = get_args()
    dataset_name = args.dataset
    if dataset_name == "None":
        print_red("error: please specify the dataset that you want to process!", file=sys.stderr)
        exit(errno.ENOENT)

    dataset_path = os.path.join(current_dir, "datasets")
    dataset_dir = os.listdir(dataset_path)
    if dataset_name not in dataset_dir:
        print_red(f"error: dataset {dataset_name} not found!", file=sys.stderr)
        exit(errno.ENOENT)

def check_feature_exist_and_generate(dataset_name):
    print_green("Checking features...")
    dataset_path = os.path.join(os.getcwd(), "datasets", dataset_name)
    class_name_list = get_class_name(dataset_path)
    if_exist_feature = isexist_feature(dataset_name)

    if if_exist_feature:
        print_green("Features already exist!")
        print_red("Warning: it will take a long time to regenerate features!(about 20 hours or more)")
        print_green("Do you want to regenerate features? (y/n)")
        choice = input()
        if choice == "y":
            sp.run(f"rm -rf npy_data/{dataset_name}", shell=True)
            generate_features(class_name_list,dataset_name)
    else:
        # print_green("features may be need a long time to generate!(about 20 hours or more)")
        generate_features(class_name_list,dataset_name)

    print_green("Features done!")

def check_result_exist_and_generate(dataset_name):
    current_dir = os.getcwd()
    result_path = os.path.join(current_dir, "result")
    
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    #检查是否有result/dataset文件夹
    result_dataset_path = os.path.join(result_path, dataset_name)
    if not os.path.exists(result_dataset_path):
        os.mkdir(result_dataset_path)
        os.mkdir(os.path.join(result_dataset_path, "cluster"))
        os.mkdir(os.path.join(result_dataset_path, "svm"))
        os.mkdir(os.path.join(result_dataset_path, "visualize"))

def main():
    print_title()   

    print_green("Checking dataset...")
    
    check_dir_exist()
    print_green("Dataset found!")
    
    #检查是否有features
    args = get_args()
    dataset_name = args.dataset
    check_feature_exist_and_generate(args.dataset)

    #检查是否有result/dataset文件夹
    check_result_exist_and_generate(args.dataset)

    #进行聚类
    print_green("Do you want to cluster? (y/n)")
    if input() in ["y", "Y"]:
        print_green("Clustering...")
        p1 = sp.run(f"python cluster.py -d {dataset_name}",shell=True)
        #如过有错误，就退出
        if p1.returncode != 0:
            exit(errno.ENOENT)
        
        #等待聚类完成
        print_green("Clustering done!")

    #进行svm分类
    print_green("Classifying...")
    p2 = sp.run(f"python classfiy.py -d {dataset_name}",shell=True)
    if p2.returncode != 0:
        exit(errno.ENOENT)
        
    print_green("Classifying done!")
    print_green("all result is in ./result/dataset_name")

    
if __name__ == "__main__":
    main()
    