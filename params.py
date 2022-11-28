# Choose your parameter
which_data = "coco2014" #coco2014, flickr8k, flickr30k 
start_from_scratch = False # False if you want to continue from the last checkpoint.
which_model = "resnet101" # resnet101, resnet152

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

# Working path
path = '/scratch/ps4534/ml/data/'

# COCO2014, Flickr8K, Flickr30K parameters
coco2014_folder = path + 'processed_HDF5/COCO2014/'
coco2014_name = 'coco2014_5_cap_per_img_5_min_word_freq'
coco2014_checkpoint = path + "checkpoints/checkpoint_" + which_model + "_" + coco2014_name + ".pth.tar"
coco2014_best_checkpoint = path + "checkpoints/BEST_checkpoint_" + which_model + "_" + coco2014_name + ".pth.tar"
coco2014_train_log = path + "logs/train_log_" + which_model + "_" + coco2014_name + ".csv"
coco2014_val_log = path + "logs/val_log_" + which_model + "_" + coco2014_name + ".csv"
coco2014_word_map_file = coco2014_folder + 'WORDMAP_' + coco2014_name + '.json' 
coco2014_train_mean = [0.4702, 0.4470, 0.4078]
coco2014_train_std = [0.2701, 0.2655, 0.2809]
coco2014_val_mean = [0.4688, 0.4453, 0.4054]
coco2014_val_std = [0.2691, 0.2642, 0.2802]
coco2014_test_mean = [0.4671, 0.4444, 0.4057]
coco2014_test_std = [0.2705, 0.2663, 0.2814]

flickr8k_folder = path + 'processed_HDF5/Flickr8k/' 
flickr8k_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
flickr8k_checkpoint = path + "checkpoints/checkpoint_" + which_model + "_" + flickr8k_name + ".pth.tar"
flickr8k_best_checkpoint = path + "checkpoints/BEST_checkpoint_" + which_model + "_" + flickr8k_name + ".pth.tar"
flickr8k_train_log = path + "logs/train_log_" + which_model + "_" + flickr8k_name + ".csv"
flickr8k_val_log = path + "logs/val_log_" + which_model + "_" + flickr8k_name + ".csv"
flickr8k_word_map_file = flickr8k_folder + 'WORDMAP_' + flickr8k_name + '.json' 
flickr8k_train_mean = [0.4580, 0.4464, 0.4032]
flickr8k_train_std = [0.2704, 0.2630, 0.2776]
flickr8k_val_mean = [0.4573, 0.4448, 0.4060]
flickr8k_val_std = [0.2713, 0.2647, 0.2799]
flickr8k_test_mean = [0.4615, 0.4501, 0.4107]
flickr8k_test_std = [0.2699, 0.2637, 0.2809]

flickr30k_folder = path + 'processed_HDF5/Flickr30k/' 
flickr30k_name = 'flickr30k_5_cap_per_img_5_min_word_freq'
flickr30k_checkpoint = path + "checkpoints/checkpoint_" + which_model + "_" + flickr30k_name + ".pth.tar"
flickr30k_log = path + "logs/logs_" + which_model + "_" + flickr30k_name + ".pth.tar"

# Seclect one from the above dataset
if which_data == "flickr8k":
    data_folder = flickr8k_folder  # folder with data files saved by create_input_files.py
    data_name = flickr8k_name  # base name shared by data files
    data_checkpoint = flickr8k_checkpoint # folder with checkpoint files saved during training
    data_best_checkpoint = flickr8k_best_checkpoint
    data_word_map_file = flickr8k_word_map_file # word map, ensure it's the same the data was encoded with and the model was trained with
    data_train_log = flickr8k_train_log
    data_val_log = flickr8k_val_log
    data_train_mean = flickr8k_train_mean
    data_train_std = flickr8k_train_std
    data_val_mean = flickr8k_val_mean
    data_val_std = flickr8k_val_std
    data_test_mean = flickr8k_test_mean
    data_test_std = flickr8k_test_std
elif which_data == "flickr30k":
    data_folder = flickr30k_folder
    data_name = flickr30k_name 
    data_checkpoint = flickr30k_checkpoint
    data_log = flickr30k_log
elif which_data == "coco2014":
    data_folder = coco2014_folder
    data_name = coco2014_name 
    data_checkpoint = coco2014_checkpoint
    data_best_checkpoint = coco2014_best_checkpoint
    data_word_map_file = coco2014_word_map_file
    data_train_log = coco2014_train_log
    data_val_log = coco2014_val_log
    data_train_mean = coco2014_train_mean
    data_train_std = coco2014_train_std
    data_test_mean = coco2014_test_mean
    data_test_std = coco2014_test_std
else:
    print(f"User selected {which_data} dataset not found.\r\nPlease select one of the available dataset (flickr8k, flickr30k, and coco2014) correctly.")
    exit()


