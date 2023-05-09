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
    parser.add_argument('--dice_threshold',
                        default=0.0,
                        type=float)
    parser.add_argument('--print_path',
                        default='',
                        type=str)
    args = parser.parse_args()
    return args


def cal_dice(pred_arr, label_arr):
    dices = []
    _sum = 5
    for i in range(1, _sum+1):
        v = ((pred_arr==i) * (label_arr==i)).sum()
        s = (pred_arr==i).sum() + (label_arr==i).sum()
        dice = 2*v/s
        dices.append(dice)
    return dices

def multiprocess_pipe(input):
    p_f, l_f = input
    pred_arr = (sitk.GetArrayFromImage(sitk.ReadImage(p_f))).astype("uint8")
    label_arr = (sitk.GetArrayFromImage(sitk.ReadImage(l_f))).astype("uint8")
    dices = cal_dice(pred_arr, label_arr)
    return dices


if __name__ == "__main__":
    args = parse_args()
    dice_threshold = args.dice_threshold
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
    for dices in result:
        if (np.array(dices)>dice_threshold).all():
            right_count+=1
    if print_path != "":
        f = open(print_path, 'a+')  
        print("Total: %d Fail: %d 合格率: %.3f"%(len(result), len(result)-right_count, right_count/len(result)), file=f)
        f.close()
    print("Total: %d Fail: %d 合格率: %.3f"%(len(result), len(result)-right_count, right_count/len(result)))
    dices_1 = [sample[0] for sample in result]
    dices_2 = [sample[1] for sample in result]
    dices_3 = [sample[2] for sample in result]
    dices_4 = [sample[3] for sample in result]
    dices_5 = [sample[4] for sample in result]
    res = pd.DataFrame(np.array([pids,dices_1,dices_2,dices_3,dices_4,dices_5]).T)
    res.to_csv(output_path,index=False,header=["pid","p1_dice","p2_dice","p3_dice","p4_dice","p5_dice"])
