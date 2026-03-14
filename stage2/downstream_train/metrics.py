import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.distance import directed_hausdorff
# 控制是否包含背景类
start = 0

# ===================== 严格对齐的Evaluator =====================
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.float64)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc if not np.isnan(Acc) else 0.0

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nan_to_num(Acc, nan=0.0)
        return np.mean(Acc)

    def Mean_Intersection_over_Union(self):
        ious = self.Intersection_over_Union()
        return np.mean(ious)

    def Intersection_over_Union(self):
        # 严格的IoU公式
        diag = np.diag(self.confusion_matrix)
        row_sum = self.confusion_matrix.sum(axis=1)
        col_sum = self.confusion_matrix.sum(axis=0)
        denominator = row_sum + col_sum - diag
        ious = np.where(denominator != 0, diag / denominator, 0.0)
        return ious

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        iu = self.Intersection_over_Union()
        freq = np.nan_to_num(freq, nan=0.0)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Dice_Score(self):
        dice_scores = {}
        for i in range(self.num_class):
            tp = np.diag(self.confusion_matrix)[i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            denominator = 2 * tp + fp + fn
            dice = 2 * tp / denominator if denominator != 0 else 0.0
            dice_scores[i] = dice
        mean_dice = np.mean(list(dice_scores.values()))
        return mean_dice, dice_scores

    def _generate_matrix(self, gt_image, pre_image):
        """生成混淆矩阵"""
        # 展平为1D数组
        gt_image = gt_image.reshape(-1)
        pre_image = pre_image.reshape(-1)
        # 仅保留有效像素
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        gt_valid = gt_image[mask]
        pre_valid = pre_image[mask]
        # 计算混淆矩阵
        label = self.num_class * gt_valid.astype(int) + pre_valid
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.float64)


# ===================== AJI和Hausdorff（支持双版本） =====================
def Hausdorff_distance(y_true, y_pred, num_classes, include_background=True):
    """
    计算Hausdorff距离
    :param include_background: 是否包含背景类
    """
    class_hd = []
    start_idx = 0 if include_background else 1
    _, y_true_idx = torch.max(y_true, 1)
    _, y_pred_idx = torch.max(y_pred, 1)

    for i in range(start_idx, num_classes):
        true = (y_true_idx == i).squeeze(0).cpu().numpy()
        pred = (y_pred_idx == i).squeeze(0).cpu().numpy()

        if np.sum(true) == 0 and np.sum(pred) == 0:
            hd = 0.0
        elif np.sum(true) == 0 or np.sum(pred) == 0:
            hd = 0.0
        else:
            hd1 = directed_hausdorff(true, pred)[0]
            hd2 = directed_hausdorff(pred, true)[0]
            hd = max(hd1, hd2)
        class_hd.append(hd)
    return class_hd


def Aggregated_jaccard_index(gt_map, predicted_map, gpu, include_background=True):
    """
    计算AJI
    :param include_background: 是否包含背景类
    """
    _, gt_map = torch.max(gt_map, 1)
    _, predicted_map = torch.max(predicted_map, 1)

    gt_list = torch.unique(gt_map)
    pr_list = torch.unique(predicted_map)

    # 是否排除背景类
    if not include_background and 0 in gt_list:
        gt_list = gt_list[gt_list != 0]
    if not include_background and 0 in pr_list:
        pr_list = pr_list[pr_list != 0]

    if len(gt_list) == 0 and len(pr_list) == 0:
        return 1.0

    pr_list = torch.cat((pr_list.view(-1, 1), torch.zeros(pr_list.size(0), 1).to(gpu)), dim=1)

    overall_correct_count = 0.0
    union_pixel_count = 0.0

    i = len(gt_list)
    while len(gt_list) > 0:
        gt = (gt_map == gt_list[i - 1]).float()
        predicted_match = gt * predicted_map.float()

        if predicted_match.sum() == 0:
            union_pixel_count += gt.sum()
            gt_list = gt_list[:-1]
            i = len(gt_list)
        else:
            predicted_nuc_index = torch.unique(predicted_match)

            if not include_background and 0 in predicted_nuc_index:
                predicted_nuc_index = predicted_nuc_index[predicted_nuc_index != 0]

            JI = 0
            best_match = None
            for j in range(len(predicted_nuc_index)):
                matched = (predicted_map == predicted_nuc_index[j]).float()
                intersection = matched.logical_and(gt).sum()
                union = matched.logical_or(gt).sum()
                nJI = intersection / union if union != 0 else 0.0
                if nJI > JI:
                    best_match = predicted_nuc_index[j]
                    JI = nJI

            if best_match is not None:
                predicted_nuclei = (predicted_map == best_match).float()
                overall_correct_count += (gt.logical_and(predicted_nuclei)).sum()
                union_pixel_count += (gt.logical_or(predicted_nuclei)).sum()

            gt_list = gt_list[:-1]
            i = len(gt_list)

            if best_match is not None:
                best_match_idx = (pr_list[:, 0] == best_match).nonzero().item()
                pr_list[best_match_idx, 1] += 1

    unused_nuclei_list = (pr_list[:, 1] == 0).nonzero().view(-1)
    for k in range(len(unused_nuclei_list)):
        unused_nuclei = (predicted_map == pr_list[unused_nuclei_list[k], 0]).float()
        union_pixel_count += unused_nuclei.sum()

    if union_pixel_count == 0:
        return 1.0
    aji = overall_correct_count / union_pixel_count
    return aji.cpu().numpy()