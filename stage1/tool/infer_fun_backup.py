import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms

def infer(model, dataroot, n_class, args):
    model.eval()
    model = model.cuda()
    cam_list = []
    gt_list = []
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, 'img/'), transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False)
    if args.dataset == 'luad':
        thr = 0.3
        # thr = 0.3
    elif args.dataset == 'bcss':
        thr = 0.5
        # thr = 0.7
    elif args.dataset == 'wsss':
        thr = 0.3
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]
        img_path = os.path.join(dataroot + 'img/' + img_name + '.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]
        def _work(i, img, thr=thr):
            with torch.no_grad():
                img = img.cuda()
                cam1, cam2, cam3, y = model.forward_cam(img)
                y = y.cpu().detach().numpy().tolist()[0]
                label = torch.tensor([1.0 if j > thr else 0.0 for j in y])
                cam3 = F.interpolate(cam3, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam1 = F.interpolate(cam1, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam2 = F.interpolate(cam2, orig_img_size, mode='bilinear', align_corners=False)[0]
                if args.dataset == 'luad':
                    cam = 0.47 * cam1 + 0.06 * cam2 + 0.47 * cam3
                if args.dataset == 'bcss':
                    cam = 0.11 * cam1 + 0.78 * cam2 + 0.11 * cam3
                if args.dataset == 'wsss':
                    cam = 0.47 * cam1 + 0.06 * cam2 + 0.47 * cam3

                cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                return cam, label
        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        # norm_cam = (sum_cam - np.min(sum_cam)) / (np.max(sum_cam) - np.min(sum_cam))
        # 替换原有的 norm_cam 计算行
        sum_cam_min = np.min(sum_cam)
        sum_cam_max = np.max(sum_cam)
        # 处理最大值和最小值相等的情况（分母为0）
        if np.isclose(sum_cam_max, sum_cam_min, atol=1e-8):
            norm_cam = np.zeros_like(sum_cam)  # 全零处理
        else:
            norm_cam = (sum_cam - sum_cam_min) / (sum_cam_max - sum_cam_min)
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img)
        bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        if iter % 100 == 0:
            print(iter)
        cam_list.append(seg_map)
        gt_map_path = os.path.join(os.path.join(dataroot, 'mask/'), img_name + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)
    return iouutils.scores(gt_list, cam_list, n_class=n_class)

def get_mask(model, dataroot, args, save_path):
    os.makedirs(save_path, exist_ok=True)  # 关键行：自动创建目录，exist_ok=True避免已存在时报错
    if args.dataset == 'luad':
        palette = [0] * 15
        palette[0:3] = [205, 51, 51]
        palette[3:6] = [0, 255, 0]
        palette[6:9] = [65, 105, 225]
        palette[9:12] = [255, 165, 0]
        palette[12:15] = [255, 255, 255]
        thr = 0.31
    elif args.dataset == 'bcss':
        palette = [0] * 15
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0, 255, 0]
        palette[6:9] = [0, 0, 255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
        thr = 0.3
    elif args.dataset == 'wsss':
        palette = [0] * 12
        palette[0:3] = [255, 255, 255]
        palette[3:6] = [0, 64, 128]
        palette[6:9] = [64, 128, 0]
        palette[9:12] = [243, 152, 0]
        thr = 0.31
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    #infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, 'img/'), transform=transform)
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot), transform=transform)
    #print(f"数据集路径：{os.path.join(dataroot)}")  # 查看是否有此输出，及路径是否正确
    infer_data_loader = DataLoader(infer_dataset,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=False)
    model = model.cuda()

    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]
        img_path = os.path.join(dataroot + img_name + '.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]
        def _work(i, img, thr=thr):
            with torch.no_grad():
                img = img.cuda()  
                cam1, cam2, cam3, y = model.forward_cam(img)
                y = y.cpu().detach().numpy().tolist()[0]
                label = torch.tensor([1.0 if j > thr else 0.0 for j in y])
                cam3 = F.interpolate(cam3, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam1 = F.interpolate(cam1, orig_img_size, mode='bilinear', align_corners=False)[0]
                cam2 = F.interpolate(cam2, orig_img_size, mode='bilinear', align_corners=False)[0]
                if args.dataset == 'luad':
                    cam = 0.47 * cam1 + 0.05 * cam2 + 0.47 * cam3
                if args.dataset == 'bcss':
                    cam = 0.11 * cam1 + 0.78 * cam2 + 0.11 * cam3
                if args.dataset == 'wsss':
                    cam = 0.47 * cam1 + 0.06 * cam2 + 0.47 * cam3
                cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                return cam, label
        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam - np.min(sum_cam)) / (np.max(sum_cam) - np.min(sum_cam))
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img)
        bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        seg_map = infer_utils.cam_npy_to_label_map(bgcam_score)
        visualimg = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        #visualimg.save(os.path.join(save_path, img_name + '.png'), format='PNG')
        save_file = os.path.join(save_path, img_name + '.png')
        print(f"保存mask到：{save_file}")  # 查看是否有此输出，及路径是否正确
        visualimg.save(save_file, format='PNG')
        if iter % 100 == 0:
            print(iter)


