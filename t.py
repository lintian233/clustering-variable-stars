import concurrent.futures

def worker(func,arg):
    return func(arg)

def add(x):
    return x+1


def my_function():
    # 线程函数的逻辑
    return "Hello, World!"

# 创建线程池
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 提交线程函数到线程池
    add1 = executor.submit(worker, add, 1)
    add2 = executor.submit(worker, add, 2)

    # 获取线程函数的返回值
    print(add1.result())
    print(add2.result())