import cv2
import numpy as np
import random
import string
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont


def generate_key(string, height, width):
    img = np.zeros((height, width, 3), np.uint8)
    img.fill(255)
    img = Image.fromarray(img)

    # create a font object with the desired size and encoding
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 40, encoding="utf-8")

    draw = ImageDraw.Draw(img)
    # 定义文字颜色
    text_color = (0, 0, 0)
    # 定义文字内容
    text = string
    # 定义文字起始位置
    left, top = 0, 0
    # 定义每行最大长度
    max_width = img.width
    # 定义空格符
    space = " "
    # 初始化一个空列表，用于存储分行后的文字
    lines = []
    # 初始化一个空字符串，用于拼接单词
    line = ""
    # 遍历每个字符
    for char in text:
        # 使用textbbox方法计算字符串的区域
        box = draw.textbbox((left, top), line + char, font=font)
        # 如果区域的宽度超过了最大长度，并且当前字符为空格符，说明可以在这里换行
        if box[2] > max_width:
            # 将字符串添加到列表中，并清空字符串
            lines.append(line)
            line = "" + char
            continue
        else:
            line += char
    # 将最后的字符串添加到列表中
    lines.append(line)
    # 遍历分行后的列表
    for line in lines:
        # 在图片上绘制一行文字，并更新起始位置
        draw.text((left, top), line, text_color, font=font)
        box = draw.textbbox((left, top), line, font=font)
        top = box[3] + 10

    img = img.resize((300, 300))
    # 保存图片
    img.save("secret.jpg")


# 定义字符集
chars = string.ascii_letters + string.digits
# 定义字符串长度
length = 512
# 生成随机字符串
random_str = "".join(random.choice(chars) for _ in range(length))
# 打印结果
print(random_str)
generate_key(random_str, 900, 900)
