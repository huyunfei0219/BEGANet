# import os
# from PIL import Image
#
#
# def convert_jpg_to_png_in_folder(source_folder, destination_folder):
#     # 检查目标文件夹是否存在，不存在则创建
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     # 遍历源文件夹中的所有文件
#     for filename in os.listdir(source_folder):
#         # 只处理 .jpg 或 .jpeg 文件
#         if filename.endswith('.jpg') or filename.endswith('.jpeg'):
#             jpg_path = os.path.join(source_folder, filename)
#             # 生成目标文件夹中的PNG文件路径
#             png_filename = filename.replace('.jpg', '.png').replace('.jpeg', '.png')
#             png_path = os.path.join(destination_folder, png_filename)
#
#             # 打开JPG图片
#             with Image.open(jpg_path) as img:
#                 # 保存为PNG格式到目标文件夹
#                 img.save(png_path, 'PNG')
#                 print(f"Converted {jpg_path} to {png_path}")
#
#
# # 示例使用
# source_folder = './eval_data/terrestrial/datatsets/TCAINet/TCAINet1000'  # 替换为原始图片文件夹的路径
# destination_folder = './eval_data/terrestrial/datatsets/TCAINet/TC1000'  # 替换为目标图片保存的文件夹路径
# convert_jpg_to_png_in_folder(source_folder, destination_folder)
#
import os
from PIL import Image

def convert_png_to_jpg_in_folder(source_folder, destination_folder):
    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 只处理 .png 文件
        if filename.endswith('.jpg'):
            png_path = os.path.join(source_folder, filename)
            # 生成目标文件夹中的JPG文件路径
            jpg_filename = filename.replace('.jpg', '.png')
            jpg_path = os.path.join(destination_folder, jpg_filename)

            # 打开PNG图片
            with Image.open(png_path) as img:
                # 如果PNG图片有透明通道（RGBA），需要转换为RGB模式
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                # 保存为JPG格式到目标文件夹
                img.save(jpg_path, 'JPEG')
                print(f"Converted {png_path} to {jpg_path}")

# 示例使用
source_folder = './eval_data/terrestrial/datatsets/ECCLNet/ECCLNet1000'  # 替换为原始图片文件夹的路径
destination_folder = './eval_data/terrestrial/datatsets/ECCLNet/ECCL1000'  # 替换为目标图片保存的文件夹路径
convert_png_to_jpg_in_folder(source_folder, destination_folder)
