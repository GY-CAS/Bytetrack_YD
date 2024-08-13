from unittest import result
from flask import Flask, jsonify, render_template, request
import json
import subprocess
import glob
import re
app = Flask(__name__)


models = {
    "0":"person",
    "1":"ship",
    "2":"uav"
}

def perform_inference(model,data_path):
    if model == models["0"]:
        # TODO: 推理代码
        # 构建命令
        command = [
            "python3", "tools/demo_track.py", "video",
            "--path", data_path, #"videos/person.mp4"
            "-f", "./exps/example/mot/yolox_tiny_mix_person_det.py",
            "-c", "./pretrained/bytetrack_tiny_mot17.pth.tar",
            "--fp16", "--fuse", "--save_result"
        ]
        try:
            # 执行命令
            subprocess.run(command, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return jsonify({"code":500,"message":"Error occurred during inference"}),500
    elif model == models["1"]:
        command = [
            "python3", "tools/demo_track.py", "video",
            "--path", data_path,
            "-f", "./exps/example/mot/yolox_tiny_Seaships_det.py",
            "-c", "./YOLOX_outputs/yolox_tiny_Seaships_det/best_ckpt.pth.tar",
            "--fp16", "--fuse", "--save_result"
        ]
        try:
            # 执行命令
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return jsonify({"code":500,"message":"Error occurred during inference"}),500

    elif model == models["2"]:
        command = [
            "python3", "tools/demo_track.py", "video",
            "--path", data_path,
            "-f", "./exps/example/mot/yolox_tiny_UAV_det.py",
            "-c", "./YOLOX_outputs/yolox_tiny_UAV_det/best_ckpt.pth.tar",
            "--fp16", "--fuse", "--save_result"
        ]
        try:
            # 执行命令
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return jsonify({"code":500,"message":"Error occurred during inference"}),500

    else:
        return jsonify({"code":400,"message":"Invalid model specified"}),400


#输入推理模型类型和推理数据路径
@app.route('/infer', methods=['POST'])
def infer():
    #request.get_json()
    model = request.json.get('infer_model')
    data_path = request.json.get('data_path')
    # 推理代码
    if model is None or model=="":
        return jsonify({"code":400,"message":"model must be specified"}),400
    if data_path is None or data_path=="":
        return jsonify({"code":400,"message":"Data Path must be specified"}),400
    # 确保模型名称有效
    if model not in models.values():
        return jsonify({'code': 400, 'message': 'Invalid model specified'}), 400

    #执行推理
    perform_inference(model,data_path)
    # 推理结束
    response = {
        "model":model,
        "data_path":data_path,
        'code': 200,
        "message":"success"
    }
    return json.dumps(response)



def get_image_filenames(folder_path):
    # 获取文件夹中的所有图片文件名
    files = glob.glob(f"{folder_path}/*.jpg")
    files.sort(key=extract_number)
    return files

def extract_number(filename):
    match = re.search(r'(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1))
    return -1

@app.route('/get_infer_results/<string:model>/', methods=['GET'])
def get_infert_results(model):
    print(model)
    #获取推理结果
    if model == models["0"]:
        #person
        img_path = "YOLOX_outputs/yolox_tiny_mix_person_det/track_vis/infer_output"
    elif model == models["1"]:
        #ship 
        img_path = "YOLOX_outputs/yolox_tiny_Seaships_det/track_vis/infer_output"

    elif model == models["2"]:
        #uav
        img_path = "YOLOX_outputs/yolox_tiny_UAV_person_det/track_vis/infer_output"


    files = get_image_filenames(img_path)
    # x  = files.pop(0)
    # files.insert(-1,x)
    return jsonify({"code":200,"message":"success","results":files})



if __name__ == '__main__':
    app.run(debug=True)