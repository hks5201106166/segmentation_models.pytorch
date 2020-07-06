#-*-coding:utf-8-*-
import shutil,os
import csv
path='/home/simple/mydemo/ocr_project/segment/data/test_results/data/image_all/'
dirs=os.listdir(path)
all_images=[]
val_csv=list(csv.reader(open('/home/simple/mydemo/ocr_project/segment/data/test_results/val.csv')))
val=[t[0] for t in val_csv]
for d in dirs:
    if d in  val:
        shutil.move(path+d,"/home/simple/mydemo/ocr_project/segment/data/test_results/data_val/" + d)

