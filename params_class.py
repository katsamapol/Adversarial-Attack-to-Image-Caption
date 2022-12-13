def return_params(which_data, which_model, target_model=""):
    param = Params(which_data, which_model, target_model)
    return [
        param.data_path,
        param.data_folder,
        param.data_name,
        param.data_checkpoint,
        param.data_best_checkpoint,
        param.data_target_best_checkpoint,
        param.data_word_map_file,
        param.data_train_log,
        param.data_val_log,
        param.data_train_mean,
        param.data_train_std,
        param.data_val_mean,
        param.data_val_std,
        param.data_test_mean,
        param.data_test_std,
        param.emb_dim,
        param.attention_dim,
        param.decoder_dim,
        param.dropout
    ]

class Params:
    def __init__(self, which_data, which_model, target_model):
        """
        Initialize parameters
        :param which_data: coco2014, flickr8k
        :param which_model: resnet18, resnet34, resnet50, resnet101, resnet152
        :param start_from_scratch: assign `True` only if you want to begin training from scratch  
        """

        # Model parameters
        self.emb_dim = 512  # dimension of word embeddings
        self.attention_dim = 512  # dimension of attention linear layers
        self.decoder_dim = 512  # dimension of decoder RNN
        self.dropout = 0.5

        # Working path
        self.data_path = "/scratch/ps4534/ml/data/"

        self.which_data = which_data
        self.which_model = which_model
        if(target_model != ""):
            self.target_model = target_model
        else:
            self.target_model = which_model
        self.data_name = f"{which_data}_5_cap_per_img_5_min_word_freq"

        self.data_best_checkpoint = f"{self.data_path}checkpoints/BEST_checkpoint_{self.which_model}_{self.data_name}.pth.tar"
        self.data_checkpoint = f"{self.data_path}checkpoints/checkpoint_{self.which_model}_{self.data_name}.pth.tar" # folder with checkpoint files saved during training
        self.data_target_best_checkpoint = f"{self.data_path}checkpoints/BEST_checkpoint_{self.target_model}_{self.data_name}.pth.tar"
        self.data_train_log = f"{self.data_path}logs/train_log_{self.which_model}_{self.data_name}.csv"
        self.data_val_log = f"{self.data_path}logs/val_log_{self.which_model}_{self.data_name}.csv"

        #if which_data == "flickr8k" or which_data == "coco2014":
        if which_data == "flickr8k":
            self.data_folder = f"{self.data_path}processed_HDF5/Flickr8k/"  # folder with data files saved by create_input_files.py
            self.data_train_mean = [0.4580, 0.4464, 0.4032]
            self.data_train_std = [0.2704, 0.2630, 0.2776]
            self.data_val_mean = [0.4573, 0.4448, 0.4060]
            self.data_val_std = [0.2713, 0.2647, 0.2799]
            self.data_test_mean = [0.4615, 0.4501, 0.4107]
            self.data_test_std = [0.2699, 0.2637, 0.2809]

        elif which_data == "coco2014":
            self.data_folder = f"{self.data_path}processed_HDF5/COCO2014/"
            self.data_train_mean = [0.4702, 0.4470, 0.4078]
            self.data_train_std = [0.2701, 0.2655, 0.2809]
            self.data_val_mean = [0.4688, 0.4453, 0.4054]
            self.data_val_std = [0.2691, 0.2642, 0.2802]
            self.data_test_mean = [0.4671, 0.4444, 0.4057]
            self.data_test_std = [0.2705, 0.2663, 0.2814]

        else:
            print(f"User selected `{which_data}` dataset not found.\r\nPlease select one of the available dataset (flickr8k, flickr30k, and coco2014) correctly.")
            exit()

        self.data_word_map_file = f"{self.data_folder}WORDMAP_{self.data_name}.json"  # word map, ensure it's the same the data was encoded with and the model was trained with

        
