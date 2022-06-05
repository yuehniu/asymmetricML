CUDA_VISIBLE_DEVICES=0 python main.py --model=resnet20 --device=gpu --batch-size=32 --dataset=imagenet --sgx

# Train cifar10 with noise
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model=vgg11_bn --device=gpu \
  --batch-size=128 \
  --dataset=cifar10 \
  --epochs=300 \
  --noisyinput \
  --nsr=0.0
