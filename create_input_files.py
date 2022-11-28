from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    # create_input_files(dataset='coco',
    #                   karpathy_json_path='/scratch/ps4534/ml/image-captioning/data/karpathy_caption_datasets/dataset_coco.json',
    #                   image_folder='/scratch/ps4534/ml/adversarial-attack-to-caption/data/images/',
    #                   captions_per_image=5,
    #                   min_word_freq=5,
    #                   output_folder='/scratch/ps4534/ml/image-captioning/data/HDF5/',
    #                   max_len=50)
    # create_input_files(dataset='flickr8k',
    #                    karpathy_json_path='/scratch/ps4534/ml/data/captions/dataset_flickr8k.json',
    #                    image_folder='/scratch/ps4534/ml/data/images/Flicker8k_Dataset/',
    #                    captions_per_image=5,
    #                    min_word_freq=5,
    #                    output_folder='/scratch/ps4534/ml/data/processed_HDF5/Flicker8k_Dataset/',
    #                    max_len=50)
    create_input_files(dataset='coco',
                      karpathy_json_path='/scratch/ps4534/ml/image-captioning/data/karpathy_caption_datasets/dataset_coco.json',
                      image_folder='/scratch/ps4534/ml/adversarial-attack-to-caption/data/images/',
                      captions_per_image=5,
                      min_word_freq=5,
                      output_folder='/scratch/ps4534/ml/data/processed_HDF5/TEST_EXPORT/',
                      max_len=50)