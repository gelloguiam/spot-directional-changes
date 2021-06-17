#python spot.py --rotate True --scale True --preprocess True --output output/output-mode-0.mp4 --mode 0
# python spot.py --output output/sample-0.mp4 --mode 0 --video './data/video/pedestrian.mp4'
# python spot.py --rotate True --scale True --preprocess True --output output/output-mode-1.mp4 --mode 1
# python spot.py --rotate True --scale True --preprocess True --output output/output-mode-2.mp4 --mode 2 --direction SE
# python spot.py --output output/xrotate-xscale-xpreprocess.mp4 --mode 1
# python spot.py --rotate True --scale True --output output/rotate-scale-xpreprocess.mp4 --mode 1


python spot.py --rotate True --scale True --preprocess True --output output-small.mp4 --mode 1 --video "data/video/physci-edited-480p.mov"
python spot.py --rotate True --preprocess True --output output-big-.mp4 --mode 1 --video "data/video/physci-full.mp4"
