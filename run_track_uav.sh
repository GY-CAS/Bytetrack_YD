python3 tools/demo_track.py video \
--path ./videos/uav-01.mp4 \
-f  ./exps/example/mot/yolox_tiny_UAV_det.py  \
-c  ./YOLOX_outputs/yolox_tiny_UAV_det/best_ckpt.pth.tar \
--fp16 --fuse --save_result