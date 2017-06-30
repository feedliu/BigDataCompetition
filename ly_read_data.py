#-*- coding:utf-8 -*-
from multiprocessing import Pool
from featureProject.ly_features import make_train_set
import time,os

def fun(x):
    if x == 31000000:
        temp_data, temp_labels = make_train_set(x, x + 1000000, sub=True)
    else: 
        temp_data, temp_labels = make_train_set(x, x + 1000000) 
    temp_data, temp_labels = None, None

if __name__ == '__main__':
    t_start = time.time()
    pool = Pool(10)
    args = [ i for i in range(18000000,31000001,1000000) ]
    pool.map(fun, args)
    pool.close(); 
    pool.join()  # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。
    pool.terminate()
    print 'the program time is :%s'%( time.time() - t_start )
