import argparse
import glob
import os
import sys
import tarfile
import traceback

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor import DetectKeypointModel, DetectKeypointPredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Test DetectKeypoint')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_dicom_path', default='../example/data/input/dcm', type=str)
    parser.add_argument('--output_path', default='../example/data/output', type=str)
    parser.add_argument(
        '--model_path',
        default=glob.glob("./data/model/*.tar")[0] if len(glob.glob("./data/model/*.tar")) > 0 else None,
        type=str,
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='../train/checkpoints/v1/model_trt.pth'
        # default=None
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./detect_keypoint.yaml'
        # default=None
    )
    args = parser.parse_args()
    return args


def inference(predictor: DetectKeypointPredictor, volume: np.ndarray):
    pred_array = predictor.predict(volume)
    return pred_array


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    sitk_img = reader.Execute()
    return sitk_img


def main(input_dicom_path, output_path, device, args):
    # TODO: 适配参数输入
    if (
        args.model_file is not None and 
        args.config_file is not None
    ):
        model_detect_keypoint = DetectKeypointModel(
            model_f=args.model_file,
            config_f=args.config_file,
        )
        predictor_detect_keypoint = DetectKeypointPredictor(
            device=device,
            model=model_detect_keypoint,
        )
    else:
        with tarfile.open(args.model_path, 'r') as tar:
            files = tar.getnames()
            model_detect_keypoint = DetectKeypointModel(
                model_f=tar.extractfile(tar.getmember('detect_keypoint.pth')),
                config_f=tar.extractfile(tar.getmember('detect_keypoint.yaml')),
            )
            predictor_detect_keypoint = DetectKeypointPredictor(
                device=device,
                model=model_detect_keypoint,
            )

    os.makedirs(output_path, exist_ok=True)
    for pid in tqdm(os.listdir(input_dicom_path)):
        pid_path = os.path.join(input_dicom_path, pid)
        if os.path.isdir(pid_path):
            sitk_img = load_scans(pid_path)
        elif pid_path.endswith(".nii.gz"):
            sitk_img = sitk.ReadImage(pid_path)
            pid = pid.replace(".nii.gz","")
        volume = sitk.GetArrayFromImage(sitk_img).astype('float32')
        pred_array = inference(predictor_detect_keypoint, volume)
        keypoint_itk = sitk.GetImageFromArray(pred_array)
        keypoint_itk.CopyInformation(sitk_img)
        sitk.WriteImage(keypoint_itk, os.path.join(output_path, f'{pid}.seg.nii.gz'))



if __name__ == '__main__':
    args = parse_args()
    main(
        input_dicom_path=args.input_dicom_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )