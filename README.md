# Adversarial-Attack-to-Image-Caption

## Requirements

## Training

Check training options: 
```
python train_args.py -h
```
To begin training, you must specify 
1. which model you want to use between resnet50, resnet101, and resnet152
2. which dataset you want to use between coco2014 and flickr8k
3. begin finetuning from scratch between True and False, select False if you want to continue training from your saved model.
4. Finetune your model encode between True and False
```
# python train_args.py --which_model="resnet101" --which_data="coco2014" --start_from_scratch="True" --fine_tune_encoder="True"

```
- Implemented Adversarial Attack using ResNet50, ResNet101, ResNet152
- We followed sgrvinod instruction on how to do image captioning
- A complete tutorial by sgrvinod can be found at https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- Tested White-Box attack on each of the Models
- Tested Black-Box attack on CLIP prefix captioning model by rmokady, can be found at https://github.com/rmokady/CLIP_prefix_caption
- 
