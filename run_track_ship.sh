python3 tools/demo_track.py video \
--path ./videos/yunshachuan3.mp4 \
-f  ./exps/example/mot/yolox_tiny_Seaships_det.py  \
-c  ./YOLOX_outputs/yolox_tiny_Seaships_det/best_ckpt.pth.tar \
--fp16 --fuse --save_result