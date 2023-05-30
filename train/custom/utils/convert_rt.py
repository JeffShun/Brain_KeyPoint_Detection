
import torch
import argparse
from torch2trt import torch2trt
import tensorrt as trt
import os, sys
work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)
from config.detect_keypoint_config import network_cfg

def load_model(model_path):
    model = network_cfg.network
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/v1/30.pth')
    parser.add_argument('--output_path', type=str, default='./checkpoints/v1')
    args = parser.parse_args()
    return args

# branch
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    model = load_model(args.model_path)
    # 定义示例输入
    x = torch.ones((1, 1, 64, 160, 160)).cuda()
    model_trt = torch2trt(model,[x], fp16_mode=True)
    torch.save(model_trt.state_dict(), os.path.join(args.output_path, 'model_trt.pth'))

