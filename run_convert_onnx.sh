############################# Person  #########################################
python3 tools/export_onnx.py \
--output-name YOLOX_outputs/yolox_tiny_mix_person_det/yolox_tiny_person.onnx \
-f exps/example/mot/yolox_tiny_mix_person_det.py \
-c YOLOX_outputs/yolox_tiny_mix_person_det/bytetrack_tiny_mot17.pth.tar

echo "Export yolox_tiny_mix_person_det onnx success to YOLOX_OUPUTS"

############################# Seaships ###################################
python3 tools/export_onnx.py \
--output-name YOLOX_outputs/yolox_tiny_Seaships_det/yolox_tiny_Seaships.onnx \
-f exps/example/mot/yolox_tiny_Seaships_det.py \
-c YOLOX_outputs/yolox_tiny_Seaships_det/best_ckpt.pth.tar

echo "Export yolox_tiny_Seaships onnx success to YOLOX_OUPUTS"

################################# UAV #######################################
python3 tools/export_onnx.py \
--output-name YOLOX_outputs/yolox_tiny_UAV_det/yolox_tiny_uav.onnx \
-f exps/example/mot/yolox_tiny_UAV_det.py \
-c YOLOX_outputs/yolox_tiny_UAV_det/best_ckpt.pth.tar

echo "Export yolox_tiny_mix_person_det onnx success to YOLOX_OUPUTS"