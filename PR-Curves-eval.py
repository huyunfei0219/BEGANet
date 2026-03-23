
import os
import ntpath
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from measures.MAE import get_MAE
from measures.S_Measure import get_SMeasure
from measures.F_Measure import get_PR_uint8, get_wFMeasure

# 配置参数
eval_dir = 'eval_data/terrestrial/'
dataset = 'datatsets'
eval_res = (224, 224)

#各方法及其文件扩展名和子目录
methods_info = {
    'ADF': ('ADF821', '.png'),
    'MTMR': ('MTMR821', '.png'),
    'SGDL': ('SGDL821', '.png'),
    'Swin': ('Swin821', '.png'),
    #'MSEDNet': ('MSEDNet821', '.png'),
   # 'EFCRFNet': ('EFCRFNet821', '.png'),
    'LSNet': ('LSNet821', '.png'),
    'AFNet':('AFNet821','.png'),
    'CPD':('CPD821','.png'),
    'FMCF':('FMCF821','.png'),
    'JLDCF':('JLDCF821','.png'),
    'MIED':('MIED821','.png'),
    'PDNet':('PDNet821','.png'),
    'R3Net':('R3Net821','.png'),
    'S2MA':('S2MA821','.png'),
    'M3S-NIR':('M3S-NIR821','.png'),
    'TANet':('TANet821','.png'),
    #'TCAINet':('TCAINet821','.png'),
    #'SwinMcNet':('SwinMcNet821','.png'),
    #'UidefNet':('UidefNet821','.png'),
    #'HRTransNet': ('HRTransNet821', '.png'),
    'ECCLNet':('ECCL821','.png'),
    'PoolNet':('PoolNet821','.png')

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
#     #'TCAINet': ('TCAINet1000', '.jpg'),
#     #'SwinMcNet': ('SwinMcNet1000', '.png'),
#     #'UidefNet': ('UidefNet1000', '.png'),
#     'M3S-NIR':('M3S-NIR1000','.png'),
#     #'HRTransNet': ('HRTransNet1000', '.png'),
#     'TANet':('TANet1000','.png'),
#     'ECCLNet':('ECCL1000','.png'),
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
#     #'HRTransNet': ('HRTransNet5000', '.png'),
#     'CPD':('CPD5000','.png'),
#     'FMCF':('FMCF5000','.png'),
#     'JLDCF':('JLDCF5000','.png'),
#     'MIED':('MIED5000','.png'),
#     'PDNet':('PDNet5000','.png'),
#     'R3Net':('R3Net5000','.png'),
#     #'TCAINet':('TCAINet5000','.png'),
#     #'SwinMcNet':('SwinMcNet5000','.png'),
#     #'UidefNet':('UidefNet5000','.png'),
#     'S2MA':('S2MA5000','.png'),
#     'M3S-NIR':('M3S-NIR5000','.png'),
#     'TANet':('TANet5000','.png'),
#     'ECCLNet':('ECCLNet5000','.png'),
#     'PoolNet':('PoolNet5000','.png')
#
# }
# 目录和路径设置
gt_dir = os.path.join(eval_dir, dataset, 'VT821', 'VT821','GT')
gt_paths = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir)) if os.path.isfile(os.path.join(gt_dir, f))]

def read_mask(path, res=(224, 224)):
    # 读取图像并返回数组 (0, 255)
    img = Image.open(path).resize(res)
    return np.array(img.convert("L"))

def compute_pr_curve(gt_paths, methods_info, eval_res):
    pr_vals = {}
    for method, (subdir, ext) in methods_info.items():
        out_dir = os.path.join(eval_dir, dataset, method, subdir)
        out_paths = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(ext)]

        all_p, all_r = [], []
        for gt_path in gt_paths:
            gt = read_mask(gt_path, res=eval_res)  # [0, 255]
            im_name = ntpath.basename(gt_path).split('.')[0]

            if method == 'ADF':
                smap_name = os.path.join(out_dir, im_name + ext)
            elif method in ['SGDL', 'MTMR']:
                smap_name = os.path.join(out_dir, f"{im_name}_stage2{ext}")
            elif method in ['M3S-NIR']:
                smap_name = os.path.join(out_dir, f"{im_name}_stage1{ext}")
            else:
                smap_name = os.path.join(out_dir, im_name + ext)

            if not os.path.isfile(smap_name):
                print(f"File not found: {smap_name}")
                continue

            smap = read_mask(smap_name, res=eval_res)  # [0, 255]
            Ps, Rs, _ = get_PR_uint8(smap, gt)
            all_p.append(Ps)
            all_r.append(Rs)

        if not all_p or not all_r:
            print(f"No data for method {method}")
            continue

        Ps = np.mean(np.array(all_p), 0)
        Rs = np.mean(np.array(all_r), 0)
        pr_vals[method] = (Rs, Ps)
    return pr_vals

def compute_fmeasure_curve(gt_paths, methods_info, eval_res):
    fmeasure_vals = {}
    bins = np.arange(0, 256)
    for method, (subdir, ext) in methods_info.items():
        out_dir = os.path.join(eval_dir, dataset, method, subdir)
        out_paths = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(ext)]

        all_fmeasure = []
        for gt_path in gt_paths:
            gt = read_mask(gt_path, res=eval_res)  # [0, 255]
            im_name = ntpath.basename(gt_path).split('.')[0]

            if method == 'ADF':
                smap_name = os.path.join(out_dir, im_name + ext)
            elif method in ['SGDL', 'MTMR']:
                smap_name = os.path.join(out_dir, f"{im_name}_stage2{ext}")
            elif method in ['M3S-NIR']:
                smap_name = os.path.join(out_dir, f"{im_name}_stage1{ext}")
            else:
                smap_name = os.path.join(out_dir, im_name + ext)

            if not os.path.isfile(smap_name):
                print(f"File not found: {smap_name}")
                continue

            smap = read_mask(smap_name, res=eval_res)  # [0, 255]
            Ps, Rs, ious = get_PR_uint8(smap, gt)
            F1 = get_wFMeasure(Ps, Rs)
            all_fmeasure.append(F1)

        if not all_fmeasure:
            print(f"No data for method {method}")
            continue

        F1 = np.mean(np.array(all_fmeasure), 0)
        fmeasure_vals[method] = F1
    return fmeasure_vals, bins

# 计算 PR 曲线数据
pr_vals = compute_pr_curve(gt_paths, methods_info, eval_res)

# 绘制 PR 曲线
plt.figure(figsize=(12, 6))
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown', 'pink', 'lightblue', 'lime', 'gold', 'navy', 'teal', 'coral', 'violet']
linestyles = ['-', '--']  # 添加虚线样式

for i, (method, (Rs, Ps)) in enumerate(pr_vals.items()):
    plt.plot(Rs, Ps, linestyle=linestyles[i % len(linestyles)], color=colors[i % len(colors)], linewidth=2, label=method)

plt.grid(True)
_font_size_ = 16
plt.title(f'PR Curve for VT821', fontsize=_font_size_ + 2)
plt.xlim([0.55, 1.0])
plt.xlabel("Recall", fontsize=_font_size_)
plt.xticks(fontsize=_font_size_ - 4)
plt.ylabel("Precision", fontsize=_font_size_)
plt.yticks(fontsize=_font_size_ - 4)
plt.legend(loc='lower left', fontsize=_font_size_ - 2, framealpha=0.75, ncol=2)
plt.savefig('ECCL821.png', bbox_inches='tight')
plt.show()


# 计算 F-measure 数据
fmeasure_vals, bins = compute_fmeasure_curve(gt_paths, methods_info, eval_res)

# 绘制 F-measure 曲线
plt.figure(figsize=(12, 6))
linestyles = ['-', '--']
for i, (method, F1) in enumerate(fmeasure_vals.items()):
    linestyle = linestyles[i % len(linestyles)]
    color = colors[i % len(colors)]
    plt.plot(bins[:-1] / 255.0, F1, linestyle=linestyle, color=color, linewidth=2, label=method)

plt.grid(True)
plt.title(f'F-measure Curve for VT821', fontsize=_font_size_ + 2)
plt.xlim([0.0, 1.0])
plt.xlabel("Threshold", fontsize=_font_size_)
plt.xticks(fontsize=_font_size_ - 4)
plt.ylabel("F-measure", fontsize=_font_size_)
plt.yticks(fontsize=_font_size_ - 4)
plt.legend(loc='lower left', fontsize=_font_size_ - 2, framealpha=0.75, ncol=2)
plt.savefig('ECCLNe821.png', bbox_inches='tight')
plt.show()
