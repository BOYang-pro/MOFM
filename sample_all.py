import os


#os.system('torchrun  --nproc_per_node=4 train.py --task VI-IR  --batch_size 12 --img_size 256')

#os.system('torchrun  --nproc_per_node=4 train.py --task VI-NIR  --batch_size 12 --img_size 256')

os.system('torchrun  --nproc_per_node=2 train.py --task Med  --batch_size 2 --img_size 256')
