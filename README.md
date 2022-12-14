# Adversarial-Attack-to-Image-Caption

## Installing Prerequisites
#TODO explaning how to install our requirements.txt using conda environment.

## Getting Data
#TODO explaning where to download COCO2014 and Flickr8K dataset 

## Data Preprocessing
#TODO explaning how to use create_input_files.py to preprocess datasets

## Training
#TODO explaning how to use train_args.py to train resnet50, resnet101, and resnet152 models
#And carify that even select "start_from_scratch" the model will still be train using pre-trained model trained on ImageNet dataset.

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
python train_args.py --which_model="resnet101" --which_data="coco2014" --start_from_scratch="True" --fine_tune_encoder="True"

```
## Evaluating
#TODO Explaning how to use eval_args.py to evaluate models.

## Captioning
#TODO Explaning how to use caption_args.py to create caption from an image.

## Attacking (ResNet)
#TODO 

## Attacking (CLIP prefix captioning)
User can use adversarial images provided in the "results/perturbed-images"
Or create their own adversarial image from attack_args.py
#TODO

- Implemented Adversarial Attack using ResNet50, ResNet101, ResNet152
- We followed sgrvinod instruction on how to do image captioning
- A complete tutorial by sgrvinod can be found at https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- Tested White-Box attack on each of the Models
- Tested Black-Box attack on CLIP prefix captioning model by rmokady, can be found at https://github.com/rmokady/CLIP_prefix_caption
- 
