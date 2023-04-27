import os
import re
import SimpleITK as sitk
import numpy as np
import argparse
import pandas as pd
from multiprocessing import Pool
import cc3d
from bw_ocean.core.bw_algo import skeletonize_3d as skeletonize_3d
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

def parse_args():
    parser = argparse.ArgumentParser('output structured_report for myocardialbridge')
    parser.add_argument('--pred_path',
                        default='../example/data/output',
                        type=str)
    parser.add_argument('--cor_path',
                        default='../example/data/input/cor_seg',
                        type=str)
    parser.add_argument('--label_path',
                        default='../example/data/input/心肌桥测试集参考结果.csv',
                        type=str)
    parser.add_argument('--output_path',
                        default='../example/data/output.csv',
                        type=str)
    parser.add_argument('--print_path',
                        default='',
                        type=str)
    args = parser.parse_args()
    return args


def process_single(pred_arr, cor_arr):
    pred_color_mask = cor_arr * pred_arr
    pred_color = np.unique(pred_color_mask)
    res = set()
    for color_id in pred_color:
        if color_id != 0 and (pred_color_mask==color_id).sum()>50:
            res.add(color_bar[color_id])
    return res

def multiprocess_pipe(inputs):
    p_f, cor_f = inputs
    cor_arr = (sitk.GetArrayFromImage(sitk.ReadImage(cor_f))).astype("uint8")
    pred_arr_ = (sitk.GetArrayFromImage(sitk.ReadImage(p_f))).astype("uint8")
    pred_arr = np.zeros_like(pred_arr_)
    pred_arr[(pred_arr_>0) & (pred_arr_<10)] = 1
    result = process_single(pred_arr, cor_arr)
    result = list(result)
    result = "、".join(result)
    print(p_f.split("/")[-1].replace("-seg.nii.gz",""), result)
    return result


if __name__ == "__main__":

    color_map = {
        "rca1": [11],
        "rca2": [12],
        "rca3": [13],
        "rpda": [14],
        "rplb": [26],
        "lm": [15],
        "lad1": [16],
        "lad2": [17],
        "lad3": [18],
        "d1": [19],
        "d2": [20],
        "lcx1": [21],
        "om1": [22],
        "om2": [24],
        "lcx2": [23],
        "lpda": [25],
        "ri": [27],
        "lplb": [28],
        "other": [60],
    }
    color_bar = ["Unkown" for i in range(100)]
    for color in color_map:
        for l in color_map[color]:
            color_bar[l] = color

    args = parse_args()
    pred_path = args.pred_path
    cor_path = args.cor_path
    output_path = args.output_path
    output_dir = "/".join(output_path.split("/")[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print_path = args.print_path
    pids = [pid.replace("-seg.nii.gz","") for pid in os.listdir(pred_path)]
    new_pids = []
    pool = Pool(8)
    inputs = []   
    for pid in pids:
        p_f = os.path.join(pred_path, pid+"-seg.nii.gz")
        cor_f = os.path.join(cor_path, pid+"-seg.nii.gz")
        if os.path.exists(cor_f):
            new_pids.append(pid)
            inputs.append([p_f, cor_f])
    result = pool.map(multiprocess_pipe, inputs)
    pool.close()
    pool.join()
    result = np.array(result)
    result = np.concatenate((np.array(new_pids)[:, np.newaxis],result[:, np.newaxis]),1)
    result = dict(zip(result[:,0],result[:,1]))

    label_path = args.label_path
    f = open(label_path,"r")
    label = pd.read_csv(f)
    res_pids = label["pid"].tolist()
    res_label = label["label"].tolist()
    res_pred = []
    res_aggree = []
    n_count = 0
    right_count = 0
    TP = 0
    ATP = 0
    APP = 0
    f.close()
    for i,pid in enumerate(res_pids):
        if pid in result:
            answer = str(res_label[i])
            pred = str(result[pid])
            res_pred.append(pred)
            pred_set = set(pred.strip().split("、"))
            answer_set = set(answer.strip().split("、"))
            ignores = ["other" ,"nan", ""]
            for ignore in ignores:
                if ignore in pred_set:
                    pred_set.remove(ignore)
                if ignore in answer_set:
                    answer_set.remove(ignore) 
            TP += len(pred_set&answer_set)
            ATP += len(answer_set)  
            APP += len(pred_set)          
            aggree = pred_set == answer_set
            res_aggree.append(aggree)
            if aggree:
                right_count += 1
            n_count += 1
        else:
            res_pred.append("find not pred!")
            res_aggree.append("find not pred!")
    if print_path != "":
        f = open(print_path, 'a+')  
        print("Total: %d Fail: %d 合格率: %.3f, 召回率: %.3f, 精确率: %.3f"%(n_count, n_count-right_count, right_count/n_count, TP/ATP, TP/APP), file=f)
        f.close()
    print("Total: %d Fail: %d 合格率: %.3f 召回率: %.3f 精确率: %.3f"%(n_count, n_count-right_count, right_count/n_count,TP/ATP, TP/APP))
    res = pd.DataFrame(np.array([res_pids,res_label,res_pred,res_aggree]).T)
    res.to_csv(output_path,index=False,header=["pid","label","pred","aggree"])
