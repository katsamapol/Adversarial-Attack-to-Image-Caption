import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import os
from scipy.misc import imread, imresize
from PIL import Image
from models_extended import *
from params_class import *
from utils import *

# Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
print(f"Device: {device}")


def main():
    args = _parse_arguments()
    (data_path,
    data_folder, 
    data_name, 
    data_checkpoint, 
    data_best_checkpoint, 
    data_target_best_checkpoint,
    data_word_map_file, 
    data_train_log, 
    data_val_log, 
    data_train_mean,  
    data_train_std, 
    data_val_mean, 
    data_val_std, 
    data_test_mean, 
    data_test_std,
    emb_dim,
    attention_dim,
    decoder_dim,
    dropout )= return_params(args.which_data, args.which_model)

    # Load word map (word2ix)
    with open(data_word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    vocab_size = len(word_map)

    # Initialize model
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                    embed_dim=emb_dim,
                                    decoder_dim=decoder_dim,
                                    vocab_size=vocab_size,
                                    dropout=dropout)

    if args.which_model == "resnet101":
        encoder = ResNet101Encoder()  # I kept it as text right now but you can import model
        print("ResNet101Encoder")
    elif args.which_model == "resnet152":
        encoder = ResNet152Encoder()
        print("ResNet152Encoder")
    elif args.which_model == "resnet50":
        encoder = ResNet50Encoder()
        print("ResNet50Encoder")
    # elif args.which_model == "resnet34":
    #     encoder = ResNet34Encoder()
    #     print("ResNet34Encoder")
    # elif args.which_model == "resnet18":
    #     encoder = ResNet18Encoder()
    #     print("ResNet18Encoder")
    else:
        print(
            f"User selected {args.which_model} model not found.\r\nPlease select one of the available models ('resnet50', 'resnet101', or 'resnet152') correctly."
        ) 
        exit()

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Load model
    checkpoint = torch.load(data_best_checkpoint, map_location=str(device))
    # decoder = checkpoint['decoder']
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    # decoder = decoder.to(device)
    decoder.eval()
    # encoder = checkpoint['encoder']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    # encoder = encoder.to(device)
    encoder.eval()

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    # print(alphas.shape)
    alphas = torch.FloatTensor(alphas)
    # print(alphas.shape)
    # Visualize caption and attention of best sequence
    visualize_att(args.img, "", seq, alphas, rev_word_map, args.smooth)

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)

    prev_mean, prev_std = torch.empty(3), torch.empty(3)
    while(1):
        mean, std = img_mean_and_std(img, device)
        # print(mean, std)
        if (torch.sum(mean) == torch.sum(prev_mean) and torch.sum(std) == torch.sum(prev_std)):
            break
        else:
            prev_mean = mean
            prev_std = std

    
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        # prev_word_inds = top_k_words / vocab_size  # (s)
        # prev_word_inds = torch.div(top_k_words, vocab_size) # (s)
        prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor') # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    
    return seq, alphas

def visualize_att(image_path, output_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    # file name without extension
    plt.savefig(f'{os.path.splitext(image_path)[0]}_caption.png', bbox_inches='tight', dpi=300)
    plt.show()

def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--which_model", default="resnet101", type=str, 
    help="Which model to use 'resnet50', 'resnet101', or 'resnet152'", choices=["resnet50","resnet101", "resnet152"])
    argparser.add_argument("-d", "--which_data", default="coco2014", type=str, 
    help="Which dataset to use 'coco2014', 'flickr8k', 'flickr30k'", choices=["coco2014", "flickr8k", "flickr30k"])
    argparser.add_argument("-b", "--beam_size", default=3, type=int,
    help="Beam size at which to generate captions for evaluation", choices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    argparser.add_argument('-i', '--img', type=str, required=True, 
    help='path to image')
    argparser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    
    return argparser.parse_args()

if __name__ == "__main__":
    main()