import os, sys, pyperclip, re

#txt 文件路径
filenames = '/media/zjx/ZJX/coco/coco_linux (copy).txt'
print(filenames)


file = open(filenames,'r') #只读方式打开文件
txtdata = file.readlines()      #读取文件
file.close()

str1 = "/media/zjx/ZJX/coco"
str2 = "/"
for line in txtdata:
    """
    str = "/media/guide/Turbo/0_ADeepLearning/open_dataset/cocodataset/train2017/JPEGImages/000000000009.jpg"
    str2 = "/"
    str1 = "/media/zjx/ZJX/coco"
    result = str.split('/')
    newfilename = str1 + str2 + result[7] + str2 + result[8] + str2 + result[9]
    """
    result = line.split('/')
    newfilename = str1 + str2 + result[7] + str2 + result[8] + str2 + result[9]
    writefile = open("/media/zjx/ZJX/coco/coco1.txt", "a+")
    writefile.write(newfilename)
writefile.close()
