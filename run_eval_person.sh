python3 tools/eval.py \
-f  ./exps/example/mot/yolox_tiny_mix_person_det.py\
-c ./YOLOX_outputs/yolox_tiny_mix_person_det/bytetrack_tiny_mot17.pth.tar\
--eval_data ./Datasets/MOT17/images\
--eval_json val_half.json\
--img_path_name half