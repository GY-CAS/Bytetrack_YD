from flask import Flask, render_template,send_from_directory
import os
import itertools

app = Flask(__name__)

# 假设你的图片存储在这个目录下
IMAGE_FOLDER = 'YOLOX_outputs/yolox_tiny_mix_person_det/track_vis/infer_output'
# 获取该目录下所有图片文件的列表
image_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]

# 使用itertools.cycle来创建一个无限循环的迭代器
image_cycle = itertools.cycle(image_files)

@app.route('/')
def show_image():
    # 获取下一个图片文件名
    image_file = next(image_cycle)
    # 发送图片
    #return send_from_directory(IMAGE_FOLDER, image_file)
    return render_template('index.html', image_files=image_files)

@app.route('/next_image')
def next_image():
    # 获取下一个图片文件名，但不发送图片，仅用于AJAX请求
    next_image_file = next(image_cycle)
    return next_image_file

if __name__ == '__main__':
    app.run(debug=True)
