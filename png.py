import os
from PIL import Image

# 源文件夹和目标文件夹路径
input_folder = r'D:\fuxian\UNeXt-pytorch-main2\UNeXt-pytorch-main\inputs\ISIC2018\images'
output_folder = r'D:\fuxian\UNeXt-pytorch-main2\UNeXt-pytorch-main\inputs\ISIC2018\images2'

# 创建目标文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # 构建完整的文件路径
        jpg_path = os.path.join(input_folder, filename)
        png_path = os.path.join(output_folder, filename[:-4] + '.png')  # 替换扩展名

        # 打开 JPG 文件并转换为 PNG
        with Image.open(jpg_path) as img:
            img.save(png_path, 'PNG')

print("转换完成！")