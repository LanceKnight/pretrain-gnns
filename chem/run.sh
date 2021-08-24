
python -W ignore finetune.py --dataset '435034' --split random  --lr 0.001 --lr_scale 1 --eval_train 0 --num_layers 5 --num_kernel1_1hop 15 --num_kernel2_1hop 15 --num_kernel3_1hop 15 --num_kernel4_1hop 15 --num_kernel1_Nhop 15 --num_kernel2_Nhop 15 --num_kernel3_Nhop 15 --num_kernel4_Nhop 15 --D 2 --num_samples 1000 --predefined_kernelsets 1  --epoch 100 --batch_size 2
#python -W ignore pretrain.py --dataset '435008' --split random --epoch 100  --eval_train 1 --num_layers 1 --num_kernel1 15  --num_kernel2 15  --num_kernel3 15  --num_kernel4 15 --D 2


