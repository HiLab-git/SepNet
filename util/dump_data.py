import os
import numpy as np
### 把鼻咽癌的3D数据转存一下

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")

        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")



def dumpdata(root, dumproot, filename):
    folder = os.path.exists(root)
    count = 0
    for fn in os.listdir(root):  # fn 表示的是文件名
        count = count + 1
    print(count)
    if folder:
        for patient in os.listdir(root):
            count = count - 1
            dumppath = dumproot + patient
            mkdir(dumppath)
            for file in filename:
                filepath = root + patient + '/' + file
                dumpfile = np.load(filepath)
                np.save(dumppath + '/' + file, dumpfile)
            print("%d patients left" % count)

root = "/lyc/RTData/Original CT/RT/"
dumproot = "/lyc/RTData/3D_data/"
filename = ["Img_norm.npy", "Mask.npy", "Patient_pixel.npy"]
dumpdata(root, dumproot, filename)
