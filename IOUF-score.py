import os
import numpy as np
import ntpath
import cv2

eval_res = (256, 256)  # 评估分辨率，按需要设置
eval_dir = 'eval_data/terrestrial/'  # 评估文件夹路径
dataset = 'datatsets'  # 数据集名称

# 读取二值化的掩膜图像
def read_mask(mask_path, res=None):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if res:
        mask = cv2.resize(mask, res)
    return mask

# 计算IoU值
def get_IoU(pred_mask, gt_mask):
    # 将输入图像转换为二值图像
    pred_mask = (pred_mask > 127).astype(np.uint8)  # 大于127为前景
    gt_mask = (gt_mask > 127).astype(np.uint8)  # 大于127为前景

    # 计算交集与并集
    intersection = np.sum(pred_mask & gt_mask)  # 交集
    union = np.sum(pred_mask | gt_mask)  # 并集

    # 计算IoU
    iou = intersection / (union + 1e-6)  # 加上一个很小的数值防止除零错误
    return iou

# 计算F1-score
def get_F1_score(pred_mask, gt_mask):
    # 将输入图像转换为二值图像
    pred_mask = (pred_mask > 127).astype(np.uint8)  # 大于127为前景
    gt_mask = (gt_mask > 127).astype(np.uint8)  # 大于127为前景

    # 计算True Positives, False Positives, False Negatives
    TP = np.sum((pred_mask == 1) & (gt_mask == 1))  # 正确预测为前景
    FP = np.sum((pred_mask == 1) & (gt_mask == 0))  # 错误预测为前景
    FN = np.sum((pred_mask == 0) & (gt_mask == 1))  # 错误预测为背景

    # 计算Precision和Recall
    precision = TP / (TP + FP + 1e-6)  # 防止除零
    recall = TP / (TP + FN + 1e-6)  # 防止除零

    # 计算F1-score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1_score

# 计算各方法的IoU和F1-score
def compute_iou_and_f1(gt_paths, methods_info, eval_res, eval_dir, dataset):
    iou_vals = {}  # 用于保存IoU值
    f1_vals = {}   # 用于保存F1-score值
    for method, (subdir, ext) in methods_info.items():
        out_dir = os.path.join(eval_dir, dataset, method, subdir)
        out_paths = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(ext)]

        all_iou = []  # 用于保存所有IoU值
        all_f1 = []   # 用于保存所有F1-score值
        for gt_path in gt_paths:
            gt = read_mask(gt_path, res=eval_res)  # 读取真实图像

            im_name = ntpath.basename(gt_path).split('.')[0]

            # 适配不同方法的文件命名方式
            if method == 'ADF':
                smap_name = os.path.join(out_dir, im_name + ext)
            elif method in ['SGDL', 'MTMR']:
                smap_name = os.path.join(out_dir, f"{im_name}_stage2{ext}")
            elif method in ['M3S-NIR']:
                smap_name = os.path.join(out_dir, f"{im_name}_stage1{ext}")
            else:
                smap_name = os.path.join(out_dir, im_name + ext)

            # 检查文件是否存在
            if not os.path.isfile(smap_name):
                print(f"File not found: {smap_name}")
                continue

            smap = read_mask(smap_name, res=eval_res)  # 读取预测图像

            # 计算IoU
            iou = get_IoU(smap, gt)
            all_iou.append(iou)

            # 计算F1-score
            f1_score = get_F1_score(smap, gt)
            all_f1.append(f1_score)

        # 计算平均IoU和F1-score
        if all_iou:
            avg_iou = np.mean(np.array(all_iou))  # 平均IoU
            iou_vals[method] = avg_iou

        if all_f1:
            avg_f1 = np.mean(np.array(all_f1))  # 平均F1-score
            f1_vals[method] = avg_f1

    return iou_vals, f1_vals

# 示例输入数据
gt_paths = [os.path.join(eval_dir, dataset, 'VT821', 'VT821','GT', f) for f in os.listdir(os.path.join(eval_dir, dataset, 'VT821', 'VT821', 'GT')) if f.endswith('.jpg')]

# 各方法的配置信息
methods_info = {
    'ADF': ('ADF821', '.png'),
    'MTMR': ('MTMR821', '.png'),
    'SGDL': ('SGDL821', '.png'),
    'Swin': ('Swin821', '.png'),
    'LSNet': ('LSNet821', '.png'),
    'AFNet': ('AFNet821', '.png'),
    'CPD': ('CPD821', '.png'),
    'FMCF': ('FMCF821', '.png'),
    'JLDCF': ('JLDCF821', '.png'),
    'MIED': ('MIED821', '.png'),
    'PDNet': ('PDNet821', '.png'),
    'R3Net': ('R3Net821', '.png'),
    'S2MA': ('S2MA821', '.png'),
    'M3S-NIR': ('M3S-NIR821', '.png'),
    'TANet': ('TANet821', '.png'),
    'TCAINet': ('TCAINet821', '.png'),
    'EFCRFNet': ('EFCRFNet821', '.png'),
    'SwinMcNet': ('SwinMcNet821', '.png'),
    'UidefNet': ('UidefNet821', '.png'),
    'HRTransNet': ('HRTransNet821', '.png'),
    'PoolNet': ('PoolNet821', '.png')
}
# methods_info = {
#     'ADF': ('ADF1000', '.png'),
#     'MTMR': ('MTMR1000', '.png'),
#     'SGDL': ('SGDL1000', '.png'),
#     'Swin': ('Swin1000', '.png'),
#     #'MSEDNet': ('MSEDNet1000', '.png'),
#     #'EFCRFNet': ('EFCRFNet1000', '.png'),
#     'LSNet': ('LSNet1000', '.png'),
#     'AFNet':('AFNet1000','.png'),
#     'CPD':('CPD1000','.png'),
#     'FMCF':('FMCF1000','.png'),
#     'JLDCF':('JLDCF1000','.png'),
#     'MIED':('MIED1000','.png'),
#     'PDNet':('PDNet1000','.png'),
#     'R3Net':('R3Net1000','.png'),
#     'S2MA':('S2MA1000','.png'),
#     'TCAINet': ('TCAINet1000', '.jpg'),
#     'SwinMcNet': ('SwinMcNet1000', '.png'),
#     'UidefNet': ('UidefNet1000', '.png'),
#     'M3S-NIR':('M3S-NIR1000','.png'),
#     'HRTransNet': ('HRTransNet1000', '.png'),
#     'TANet':('TANet1000','.png'),
#    # 'MMCI':('MMCI1000','.png'),
#     'PoolNet':('PoolNet1000','.png')
#
# }
# methods_info = {
#     'ADF': ('ADF5000', '.png'),
#     'MTMR': ('MTMR5000', '.png'),
#     'SGDL': ('SGDL5000', '.png'),
#     'Swin': ('Swin5000', '.png'),
#     #'MSEDNet': ('MSEDNet5000', '.png'),
#     #'EFCRFNet': ('EFCRFNet5000', '.png'),
#     'LSNet': ('LSNet5000', '.png'),
#     'AFNet':('AFNet5000','.png'),
#     'HRTransNet': ('HRTransNet5000', '.png'),
#     'CPD':('CPD5000','.png'),
#     'FMCF':('FMCF5000','.png'),
#     'JLDCF':('JLDCF5000','.png'),
#     'MIED':('MIED5000','.png'),
#     'PDNet':('PDNet5000','.png'),
#     'R3Net':('R3Net5000','.png'),
#     'TCAINet':('TCAINet5000','.png'),
#     'SwinMcNet':('SwinMcNet5000','.png'),
#     'UidefNet':('UidefNet5000','.png'),
#     'S2MA':('S2MA5000','.png'),
#     'M3S-NIR':('M3S-NIR5000','.png'),
#     'TANet':('TANet5000','.png'),
#     #'MMCI':('MMCI5000','.png'),
#     'PoolNet':('PoolNet5000','.png')
#
# }
# 计算IoU和F1-score
iou_vals, f1_vals = compute_iou_and_f1(gt_paths, methods_info, eval_res, eval_dir, dataset)

# 打印IoU值和F1-score
print("IoU Values: ", iou_vals)
print("F1 Scores: ", f1_vals)
