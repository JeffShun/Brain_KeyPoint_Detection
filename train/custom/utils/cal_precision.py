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
    parser.add_argument('--angle_error_threshold',
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


def cal_normal_vector(image):
    point_img = sitk.GetArrayFromImage(image)
    points = []
    for i in range(1,6): 
        loc = np.array(list(zip(*np.where(point_img==i))))
        loc_center = np.mean(loc, 0)
        loc_center = tuple((loc_center + 0.5).astype(np.int64))
        points.append(loc_center)

    # 获取图像信息
    size = image.GetSize()  # 图像尺寸
    spacing = image.GetSpacing()  # 间距
    origin = image.GetOrigin()  # 原点坐标

    vtkpoints = []
    for point in points:
        pixel_z,pixel_y,pixel_x = point
        # 将像素坐标转换为 VTK 坐标
        vtk_x = origin[0] + pixel_x * spacing[0]
        vtk_y = origin[1] + pixel_y * spacing[1]
        vtk_z = origin[2] + pixel_z * spacing[2]
        vtk_point = (vtk_x,vtk_y,vtk_z)
        vtkpoints.append(vtk_point)

    point1 = vtkpoints[0]  
    point2 = vtkpoints[1]  
    point3 = vtkpoints[2] 
    point4 = vtkpoints[3]  
    point5 = vtkpoints[4]   

    # 计算矢状面的法线向量
    v1 = np.array(point5) - np.array(point1)
    v1 /= np.linalg.norm(v1)
    v2 = np.array(point3) - np.array(point1)
    v2 /= np.linalg.norm(v2)
    normal_sagittal = tuple(np.cross(v1, v2))
    normal_sagittal /= np.linalg.norm(normal_sagittal)

    # 计算横断面的法线向量
    v1 = np.array(point5) - np.array(point4)
    v1 /= np.linalg.norm(v1)
    axis1_axial = v1
    axis2_axial = normal_sagittal
    normal_axial = tuple(np.cross(axis1_axial, axis2_axial))
    normal_axial /= np.linalg.norm(normal_axial)

    # 计算冠状面的法线向量
    v1 = np.array(point3) - np.array(point2)
    v1 /= np.linalg.norm(v1)
    axis1_coronal = v1
    axis2_coronal = normal_sagittal
    normal_coronal = tuple(np.cross(axis1_coronal, axis2_coronal))
    normal_coronal /= np.linalg.norm(normal_coronal)   

    return normal_sagittal, normal_axial, normal_coronal

def cal_angle_error(pred_img, label_img):
    normal_sagittal_pred, normal_axial_pred, normal_coronal_pred = cal_normal_vector(pred_img)
    normal_sagittal_label, normal_axial_label, normal_coronal_label = cal_normal_vector(label_img)
    sagittal_angle_error = 180*np.arccos(np.dot(np.array(normal_sagittal_pred), np.array(normal_sagittal_label)))/np.pi
    axial_angle_error = 180*np.arccos(np.dot(np.array(normal_axial_pred), np.array(normal_axial_label)))/np.pi
    coronal_angle_error = 180*np.arccos(np.dot(np.array(normal_coronal_pred), np.array(normal_coronal_label)))/np.pi
    return [sagittal_angle_error, axial_angle_error, coronal_angle_error]


def multiprocess_pipe(input):
    p_f, l_f = input
    pred_img = sitk.ReadImage(p_f)
    label_img = sitk.ReadImage(l_f)
    dist = cal_dist(pred_img, label_img)
    angle_error = cal_angle_error(pred_img, label_img)
    return dist, angle_error


if __name__ == "__main__":
    args = parse_args()
    dist_threshold = args.dist_threshold
    angle_error_threshold = args.angle_error_threshold
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
    for dist, angle_error in result:
        if (np.array(dist) < dist_threshold).all() and (np.array(angle_error) < angle_error_threshold).all():
            right_count+=1
    if print_path != "":
        f = open(print_path, 'a+')  
        print("Total: %d Fail: %d 合格率: %.3f"%(len(result), len(result)-right_count, right_count/len(result)), file=f)
        f.close()
    print("Total: %d Fail: %d 合格率: %.3f"%(len(result), len(result)-right_count, right_count/len(result)))
    dist_1 = [sample[0][0] for sample in result]
    dist_2 = [sample[0][1] for sample in result]
    dist_3 = [sample[0][2] for sample in result]
    dist_4 = [sample[0][3] for sample in result]
    dist_5 = [sample[0][4] for sample in result]
    angle_error_1 = [sample[1][0] for sample in result]
    angle_error_2 = [sample[1][1] for sample in result]
    angle_error_3 = [sample[1][2] for sample in result]

    res = pd.DataFrame(np.array([pids,dist_1,dist_2,dist_3,dist_4,dist_5,angle_error_1,angle_error_2,angle_error_3]).T)
    res.to_csv(output_path,index=False,header=["pid","p1_error","p2_error","p3_error","p4_error","p5_error","sagittal_angle_error","axial_angle_error","coronal_angle_error"])
