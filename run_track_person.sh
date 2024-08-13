python3 tools/demo_track.py video \
--path videos/person.mp4 \
-f  ./exps/example/mot/yolox_tiny_mix_person_det.py \
-c ./pretrained/bytetrack_tiny_mot17.pth.tar \
--fp16 --fuse --save_result