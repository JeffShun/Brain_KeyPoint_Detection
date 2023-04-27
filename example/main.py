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

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--input_dicom_path', default='./data/input/dicom', type=str)
    parser.add_argument('--output_path', default='./data/output', type=str)
    parser.add_argument(
        '--model_path',
        default=glob.glob("./data/model/*.tar")[0] if len(glob.glob("./data/model/*.tar")) > 0 else None,
        type=str,
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='../train/checkpoints/v1/epoch_1.pth'
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


def main(input_dicom_path, output_path, gpu, args):
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
            gpu=gpu,
            model=model_detect_keypoint,
        )
    else:
        with tarfile.open(args.model_path, 'r') as tar:
            files = tar.getnames()
            model_detect_keypoint = DetectKeypointModel(
                model_f=tar.extractfile(tar.getmember('detect_keypoint.pt')),
                config_f=tar.extractfile(tar.getmember('detect_keypoint.yaml')),
            )
            predictor_detect_keypoint = DetectKeypointPredictor(
                gpu=gpu,
                model=model_detect_keypoint,
            )

    os.makedirs(output_path, exist_ok=True)

    for pid in tqdm(os.listdir(input_dicom_path)):
        try:
            sitk_img = load_scans(os.path.join(input_dicom_path, pid))
            volume = sitk.GetArrayFromImage(sitk_img)
        
            pred_array = inference(predictor_detect_keypoint, volume)
            keypoint_itk = sitk.GetImageFromArray(pred_array)
            keypoint_itk.CopyInformation(sitk_img)
            sitk.WriteImage(keypoint_itk, os.path.join(output_path, f'{pid}-kp.nii.gz'))
        except:  
            traceback.print_exc()
            break


if __name__ == '__main__':
    args = parse_args()
    main(
        input_dicom_path=args.input_dicom_path,
        output_path=args.output_path,
        gpu=args.gpu,
        args=args,
    )