import os
import re
import SimpleITK as sitk
import numpy as np
import argparse
import pandas as pd
from multiprocessing import Pool
import sys

def parse_args():
    parser = argparse.ArgumentParser('cal precision')
    parser.add_argument('--pred_path',
                        default='../../../example/data/output',
                        type=str)
    parser.add_argument('--label_path',
                        default='../../../example/data/input/label',
                        type=str)
    parser.add_argument('--output_path',
                        default='../../../example/data/output.csv',
                        type=str)
    parser.add_argument('--dist_threshold',
                        default=10.0,
                        type=float)
    parser.add_argument('--print_path',
                        default='',
                        type=str)
    args = parser.parse_args()
    return args


def cal_dist(pred_img, label_img):
    spacing = pred_img.GetSpacing()
    rotate_axis = [2,1,0] #rotate the spacing from x,y,z to z,y,x
    spacing = np.array(spacing)[rotate_axis]
    pred_arr = (sitk.GetArrayFromImage(pred_img)).astype("uint8")
    label_arr = (sitk.GetArrayFromImage(label_img)).astype("uint8")
    dists = []
    for i in range(1, 6):
        pred = np.array(list(zip(*np.where(pred_arr==i))))
        pred_center = np.mean(pred, 0)
        pred_center = tuple((pred_center + 0.5).astype(np.int64))
        pred_center = np.array(pred_center) * spacing 

        label = np.array(list(zip(*np.where(label_arr==i))))
        label_center = np.mean(label, 0)
        label_center = tuple((label_center + 0.5).astype(np.int64))
        label_center = np.array(label_center) * spacing         
        dist = np.sqrt(np.sum((pred_center - label_center)**2))
        dists.append(dist)
    return dists


def multiprocess_pipe(input):
    p_f, l_f = input
    pred_img = sitk.ReadImage(p_f)
    label_img = sitk.ReadImage(l_f)
    dist = cal_dist(pred_img, label_img)
    return dist


if __name__ == "__main__":
    args = parse_args()
    dist_threshold = args.dist_threshold
    label_path = args.label_path
    pred_path = args.pred_path
    print_path = args.print_path
    output_path = args.output_path
    output_dir = "/".join(output_path.split("/")[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pids = [pid.replace(".seg.nii.gz","") for pid in os.listdir(pred_path)]
    pool = Pool(8)
    inputs = []   
    for pid in pids:
        p_f = os.path.join(pred_path, pid+".seg.nii.gz")
        l_f = os.path.join(label_path, pid+".seg.nii.gz")
        inputs.append((p_f, l_f))
    result = pool.map(multiprocess_pipe, inputs)
    pool.close()
    pool.join()

    right_count = 0
    for dist in result:
        if (np.array(dist) < dist_threshold).all():
            right_count+=1
    if print_path != "":
        f = open(print_path, 'a+')  
        print("Total: %d Fail: %d 合格率: %.3f"%(len(result), len(result)-right_count, right_count/len(result)), file=f)
        f.close()
    print("Total: %d Fail: %d 合格率: %.3f"%(len(result), len(result)-right_count, right_count/len(result)))
    dist_1 = [sample[0] for sample in result]
    dist_2 = [sample[1] for sample in result]
    dist_3 = [sample[2] for sample in result]
    dist_4 = [sample[3] for sample in result]
    dist_5 = [sample[4] for sample in result]
    res = pd.DataFrame(np.array([pids,dist_1,dist_2,dist_3,dist_4,dist_5]).T)
    res.to_csv(output_path,index=False,header=["pid","p1_error","p2_error","p3_error","p4_error","p5_error"])
