python3 tools/eval.py \
-f ./exps/example/mot/yolox_tiny_Seaships_det.py\
-c ./YOLOX_outputs/yolox_tiny_Seaships_det/best_ckpt.pth.tar\
--eval_data /home/xwclient/workspace/zgy/ByteTrack/Datasets/Seaships\
--eval_json test.json\
--img_path_name test