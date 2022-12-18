import argparse
from utils import create_input_files
from params_class import *


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--which_data", default="coco2014", type=str, 
    help="Which dataset to use 'coco2014', 'flickr8k'", choices=["coco2014", "flickr8k"])

    return argparser.parse_args()
    
if __name__ == '__main__':
    # Create input files (along with word map)
    args = _parse_arguments()
    if(args.which_data == "coco2014"):
        create_input_files(dataset='coco',
                        karpathy_json_path=data_path+'caption_datasets/dataset_coco.json',
                        image_folder=data_path+'images/coco2014',
                        captions_per_image=5,
                        min_word_freq=5,
                        output_folder=data_path+'processed_HDF5/coco2014/',
                        max_len=50)
    elif(args.which_data == "flickr8k"):
        create_input_files(dataset='flickr8k',
                        karpathy_json_path=data_path+'caption_datasets/dataset_flickr8k.json',
                        image_folder=data_path+'images/flickr8k/',
                        captions_per_image=5,
                        min_word_freq=5,
                        output_folder=data_path+'processed_HDF5/flickr8k/',
                        max_len=50)