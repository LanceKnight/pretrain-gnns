# python finetune.py --dataset aldh1 --split random
# python finetune.py --dataset adrb2 --split random
# python finetune.py --dataset tox21 --split random
python finetune.py --dataset '435008' --split random --epoch 5  --eval_train 1 --num_layers 5 --num_kernelsets 15
