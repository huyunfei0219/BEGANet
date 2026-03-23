# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 1. 构造 DataFrame
# data = {
#     'Dataset': ['NLPR', 'NLPR', 'NLPR', 'NLPR',
#                 'NJU2K', 'NJU2K', 'NJU2K', 'NJU2K',
#                 'SIP',   'SIP',   'SIP',   'SIP',
#                 'DUT',   'DUT',   'DUT',   'DUT',
#                 'LFSD',  'LFSD',  'LFSD',  'LFSD'],
#     'Loss Fusion': ['BCE', 'IoU', 'BCE+IoU', 'WeightBCE'] * 5,
#     'MAE': [0.019, 0.0160, 0.016, 0.017,
#             0.027, 0.0260, 0.0250, 0.023,
#             0.039, 0.0400, 0.0380, 0.03,
#             0.022, 0.0200, 0.0190, 0.0178,
#             0.056, 0.0500, 0.0560, 0.048],
#     'Fm':  [0.903, 0.929, 0.9424, 0.927,
#             0.923, 0.930, 0.932, 0.935,
#             0.907, 0.906, 0.908, 0.936,
#             0.943, 0.953, 0.958, 0.960,
#             0.873, 0.884, 0.879, 0.888],
#     'Em':  [0.962, 0.968, 0.970, 0.971,
#             0.931, 0.932, 0.937, 0.932,
#             0.937, 0.930, 0.935, 0.952,
#             0.966, 0.967, 0.973, 0.978,
#             0.901, 0.918, 0.902, 0.926],
#     'Sm':  [0.939, 0.936, 0.940, 0.936,
#             0.935, 0.926, 0.934, 0.961,
#             0.904, 0.040, 0.898, 0.919,
#             0.949, 0.943, 0.953, 0.952,
#             0.882, 0.881, 0.884, 0.896],
# }
#
# df = pd.DataFrame(data)
#
# # 2. 指定绘图顺序
# loss_types = ['BCE', 'IoU', 'BCE+IoU', 'WeightBCE']
# datasets   = df['Dataset'].unique()
#
# # 3. 绘制 MAE 曲线
# plt.figure()
# for loss in loss_types:
#     subset = df[df['Loss Fusion'] == loss]
#     plt.plot(datasets, subset['MAE'], marker='o', label=loss)
# plt.title('MAE across Datasets')
# plt.xlabel('Dataset')
# plt.ylabel('MAE ↓')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # 4. 绘制 Fm 曲线
# plt.figure()
# for loss in loss_types:
#     subset = df[df['Loss Fusion'] == loss]
#     plt.plot(datasets, subset['Fm'], marker='o', label=loss)
# plt.title('Fm across Datasets')
# plt.xlabel('Dataset')
# plt.ylabel('Fm ↑')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # 5. 绘制 Em 曲线
# plt.figure()
# for loss in loss_types:
#     subset = df[df['Loss Fusion'] == loss]
#     plt.plot(datasets, subset['Em'], marker='o', label=loss)
# plt.title('Em across Datasets')
# plt.xlabel('Dataset')
# plt.ylabel('Em ↑')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # 6. 绘制 Sm 曲线
# plt.figure()
# for loss in loss_types:
#     subset = df[df['Loss Fusion'] == loss]
#     plt.plot(datasets, subset['Sm'], marker='o', label=loss)
# plt.title('Sm across Datasets')
# plt.xlabel('Dataset')
# plt.ylabel('Sm ↑')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 1. 构造 DataFrame
data = {
    'Dataset': ['SIP', 'SIP', 'SIP',
                'LSFD', 'LSFD', 'LSFD',
                'NLPR', 'NLPR', 'NLPR',
                'NJU2K','NJU2K','NJU2K',
                'DUT', 'DUT', 'DUT'],
    'Variant': ['No-CLRM', 'No-CE', 'ECCLNet'] * 5,
    'MAE':    [0.036, 0.043, 0.030,
               0.059, 0.0516,0.048,
               0.0216,0.0200,0.017,
               0.024, 0.0290,0.023,
               0.0246,0.0238,0.0178],
    'Fm':     [0.921, 0.904, 0.936,
               0.795, 0.879, 0.888,
               0.928, 0.917, 0.927,
               0.910, 0.921, 0.935,
               0.951, 0.944, 0.960],
    'Em':     [0.932, 0.934, 0.952,
               0.910, 0.924, 0.926,
               0.964, 0.964, 0.971,
               0.932, 0.955, 0.932,
               0.966, 0.968, 0.978],
    'Sm':     [0.889, 0.868, 0.919,
               0.785, 0.858, 0.896,
               0.922, 0.921, 0.936,
               0.957, 0.892, 0.961,
               0.932, 0.914, 0.952],
}

df = pd.DataFrame(data)

# 2. 保持数据集和变体顺序
datasets = ['SIP','LSFD','NLPR','NJU2K','DUT']
variants = ['No-CLRM', 'No-CE', 'ECCLNet']

# 3. 绘制各指标曲线的函数
def plot_metric(metric, ylabel, invert=False):
    plt.figure()
    for var in variants:
        sub = df[df['Variant'] == var]
        y = sub[metric]
        # 如果是 MAE（↓ 越小越好），可以在标题或 y 轴标签中注明
        if invert:
            plt.plot(datasets, y, marker='o', label=var)
        else:
            plt.plot(datasets, y, marker='o', label=var)
    plt.xlabel('Dataset')
    plt.ylabel(ylabel + (' ↓' if invert else ' ↑'))
    plt.title(f'{ylabel} across Variants')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 4. 调用绘制
plot_metric('MAE', 'MAE', invert=True)
plot_metric('Fm',  'Fm')
plot_metric('Em',  'Em')
plot_metric('Sm',  'Sm')
