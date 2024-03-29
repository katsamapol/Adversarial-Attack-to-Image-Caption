# Adversarial-Attack-to-Image-Caption
- This project focuses on creating adversarial examples to attack unknown image-captioning models and test their robustness.
- Our code built on python 3.6 and PyTorch 1.10. The list of dependencies are in the [environment.yml](environment.yml) file.
- The code does not working with python 3.7 and above, however, it can be refactored to work with python 3.7+ by changing the image manipulation library from [scipy.misc.imread](https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.misc.imread.html) to other newer libraries such as [imageio](https://imageio.readthedocs.io/en/v2.8.0/userapi.html). 

## Contents
- [Disclaimer](#Disclaimer)
- [Installing Prerequisites](#Installing-Prerequisites)
- [Getting Data](#Getting-Data)
- [Data Processing](#Data-Processing)
- [Training](#Training)
- [Evaluating](#Evaluating)
- [Generating Adversarial Examples](#Generating-adversarial-examples)
- [Attacking CLIP Prefix Captioning Model with the Adversarial Examples](#Attacking-CLIP-Prefix-Captioning-Model-with-the-Adversarial-Examples)

## Disclaimer
- We want to thank sgrvinod's repository [A PyTorch Totorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) for his comprehensive image-captioning tutorial.
- We want to thank rmokady's repository [CLIP Prefix Captioning](https://github.com/rmokady/CLIP_prefix_caption) for creating image-captioning model from CLIP.

Both these projects incredibly inspired us to work on this topic.

## Installing Prerequisites
Make sure you have conda installed.
- [Windows](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)
- [macOS](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html)
- [Linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)


Create new conda environment using provided [environment.yml](environment.yml)*

**Optional**: You can change environment name by editing the first line in environment.yml from `adv_caption` to your preference name.
```
conda env create -f /path_to_your_file/environment.yml file.
```

<!-- ```
conda create -n "[your_environment_name]" python=3.6 
``` -->

Then run:
```
conda activate adv_caption
```

To activate the conda environment you have just created.

<!-- After that, install [requirement.txt](https://github.com/katsamapol/Adversarial-Attack-to-Image-Caption/blob/main/requirements.txt) with conda install command.
```
conda install --file /path_to_your_file/requirements.txt
``` -->

## Getting Data
This repository supports working with MSCOCO2014 dataset and Flickr8K dataset.
- IF you choose to work with MSCOCO2014, download [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip)
<!-- , and [test](http://images.cocodataset.org/zips/test2014.zip). -->
- IF you choose to work with Flickr8k, the dataset can be requested [here](https://forms.illinois.edu/sec/1713398).
- Download captions of the images created by Andrej Karpathy and Li Fei-Fei in JSON blobs format [here](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

## Data Preprocessing
- In the fifth line of `params_class.py`, specify data path to your working directory e.g., `data_path = "/[dir_name]/data/"`
- Your Karpathy's JSON files should be extracted to the same directory i.e., `/[dir_name]/data/caption_datasets/`
- If you choose to work with MSCOCO2014, your images folder should look like `/[dir_name]/data/images/coco2014/train2014/` for train2014.zip, `/[dir_name]/data/images/coco2014/val2014/` for val2014.zip, and `/[dir_name]/data/images/coco2014/test2014/` for test2014.zip
- If you choose to work with Flickr8K, your images folder should look like `/[dir_name]/data/images/flickr8k/`

From now on, don't forget to run every command inside your conda environment with python3.6 installed.
For MSCOCO2014 dataset, run:
```python
python create_input_files.py --which_data="coco2014"
```
For Flick8K dataset, run:
```python
python create_input_files.py --which_data="flickr8k"
```

## Training
Check training options: 
```python
python train_args.py -h
```
To begin training, you must specify
1. which model you want to use between resnet50, resnet101, and resnet152.
2. which dataset you want to use between coco2014 and flickr8k.
3. begin finetuning from scratch between True and False, select False if you want to continue training from your saved model.
4. Finetune your model encode between True and False.
```python
python train_args.py --which_model="resnet101" --which_data="coco2014" --start_from_scratch="True" --fine_tune_encoder="True"

```
## Evaluating

Once you have completed training for at least one epoch, a model checkpoint will be saved at `/[dir_name]/data/checkpoints/`.

To evaluate your model, run:
```python
python eval_args.py --which_model="resnet101" --which_data="coco2014"
```

## Captioning

To generate caption of an image, run:
```python
python caption_args.py --which_model="resnet101" --which_data="coco2014" --img="[path_to_the_image]"
```
You will see the path to output image after the image has been successfully captioned.

## Generating adversarial examples

To generate adversarial examples from images in test set, run:
```python
python attack_args.py --which_model="resnet101" --target_model="resnet101" --which_data="coco2014" --epsilon=0.004 --export_caption="True" --export_original_image="True" --export_perturbed_image="True"
```

## Attacking CLIP Prefix Captioning Model with the Adversarial Examples
- If you did not use our `environment.yml` to install dependencies, you must install CLIP module and transformer module first. Before running the following command, make sure conda `adv_caption` environment is still activated.
```
pip install git+https://github.com/openai/CLIP.git
pip install transformers~=4.10.2
```
- If you work with MSCOCO dataset, download pre-trained COCO model for CLIPcap [here](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/). Place your downloaded file(s) inside `checkpoints` folder i.e., `/[dir_name]/data/checkpoints/coco_weights.pt`.
- If you work with Flickr8k dataset, download pre-trained conceptual captions for CLIPcap [here](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/). Place your downloaded file(s) inside `checkpoints` folder i.e., `/[dir_name]/data/checkpoints/conceptual_weights.pt`.
- 
After you have generated adversarial sample, installed dependencites, and downloaded pre-trained model, you can begin testing CLIPcap robustness by running:
```python
python attack_clipcap_eval.py --which_model="resnet101" --which_data="coco2014" --epsilon=0.004
```

