# Transformer_MAE parameter
python main --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --model deit_small_patch16_224 --data-path /home/lt/Desktop/LT_transformer/deit-main/data/cifar-10-python  --data-set CIFAR --output_dir /home/lt/Desktop/LT_transformer/deit-main/data/cifar-10-python

# 超参数
for CIFAR-10 and Cars we use:

Image size: 224 or 384 (it's to simplify because we don't change the patche size.)
Batch size: 768
lr: 0.01
optimizer: SGD
weight-decay: 1e-4
epochs: 1000
