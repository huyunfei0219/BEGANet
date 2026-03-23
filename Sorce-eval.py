import os
import ntpath
import numpy as np
from PIL import Image
from measures.MAE import get_MAE
from measures.S_Measure import get_SMeasure
from measures.F_Measure import get_PR_uint8, get_wFMeasure

# 配置参数
eval_dir = 'eval_data/terrestrial/'
dataset = 'datatsets'
eval_res = (224, 224)

# 各方法及其文件扩展名
methods_info = {
    'ADF': '.png',
    'SGDL': '.png',
    'MTMR': '.png',
    'Ours': '.png',
    'Swin': '.png'

}

# 目录和路径设置
gt_dir = os.path.join(eval_dir, dataset, 'VT821', 'VT821', 'GT')
gt_paths = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir)) if os.path.isfile(os.path.join(gt_dir, f))]


def read_mask(path, res=(224, 224)):
    img = Image.open(path).resize(res)
    return np.array(img.convert("L"))


def read_and_scale_mask(path, res=(224, 224), thr=0.5):
    img = read_mask(path, res) / 255.0
    return (img > thr).astype(np.int32)


# 对每种方法进行评估
for method in methods_info.keys():
    # 指定方法子目录，例如 'ASNetVT821'
    method_subdir = method + '821'  # 例如 'ASNetVT821'
    method_dir = os.path.join(eval_dir, dataset, method, method_subdir)

    # 收集 method_subdir 目录下的所有预测结果文件
    out_paths = [os.path.join(method_dir, f) for f in sorted(os.listdir(method_dir)) if
                 f.endswith(methods_info[method])]

    # 初始化指标
    MAEs, SMeasures, F1s = [], [], []
    all_p, all_r, all_iou = [], [], []

    for gt_path in gt_paths:
        gt = read_and_scale_mask(gt_path, res=eval_res)  # [0/1]
        im_name = ntpath.basename(gt_path).split('.')[0]
        smap_name = None

        # 在预测结果文件中查找与 gt_image 对应的预测文件
        for path in out_paths:
            if im_name in path:
                smap_name = path
                break

        if smap_name is None or not os.path.isfile(smap_name):
            continue

        # 计算 MAE 和 S-Measure
        smap = read_and_scale_mask(smap_name, res=eval_res)  # [0/1]
        MAEs.append(get_MAE(smap, gt))
        SMeasures.append(get_SMeasure(smap, gt))

        # 计算加权 F-Measure
        smap = read_mask(smap_name, res=eval_res)  # [0, 255]
        Ps, Rs, ious = get_PR_uint8(smap, gt)
        F1s.append(get_wFMeasure(Ps, Rs))
        all_p.append(Ps)
        all_r.append(Rs)
        all_iou.append(ious)

    F1s = np.mean(np.array(F1s), 0)
    F1s_max = np.max(F1s)

    # 打印结果
    print("\n{0} on {1} ({2} images):".format(method, dataset, len(MAEs)))
    print("-------------------------------------------------")
    print("Mean MAE: {0}".format(np.round(np.mean(MAEs), 4)))
    print("Mean S-Measure: {0}".format(np.round(np.mean(SMeasures), 4)))
    print("Mean F-Measure: {0}".format(np.round(F1s_max, 4)))
