from utils import create_input_files

if __name__ == '__main__':
    create_input_files(dataset='coco',
                      karpathy_json_path='/data/dataset_coco.json',
                      image_folder='/data/images/',
                      captions_per_image=5,
                      min_word_freq=5,
                      output_folder='/data/processed_HDF5/COCO2014/',
                      max_len=50)