
python -W ignore finetune.py --dataset '435008' --split random --epoch 100  --lr 0.01 --lr_scale 1 --eval_train 0 --num_layers 5 --num_kernel1 15  --num_kernel2 15  --num_kernel3 15  --num_kernel4 15 --D 2 --num_samples 3000
#python -W ignore pretrain.py --dataset '435008' --split random --epoch 100  --eval_train 1 --num_layers 1 --num_kernel1 15  --num_kernel2 15  --num_kernel3 15  --num_kernel4 15 --D 2


