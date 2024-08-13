python3 tools/eval.py \
-f  ./exps/example/mot/yolox_tiny_UAV_det.py\      
-c ./YOLOX_outputs/yolox_tiny_UAV_det/best_ckpt.pth.tar\
--eval_data ./Datasets/UAV_DET_VOC\
--eval_json instances_test.json\
--img_path_name test