# python finetune.py --dataset aldh1 --split random
# python finetune.py --dataset adrb2 --split random
# python finetune.py --dataset tox21 --split random
python -W ignore finetune.py --dataset '435008' --split random --epoch 1  --eval_train 1 --num_layers 1 --num_kernel1 10  --num_kernel2 10  --num_kernel3 10  --num_kernel4 10 
