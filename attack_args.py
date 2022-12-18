import argparse
import time
import torch as nn
import torch.nn.functional as F
import json
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
import os
from models_extended import *
from datasets import *
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
    dropout )= return_params(args.which_data, args.which_model, args.target_model)

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, grad_clip, alpha_c, fine_tune_encoder, word_map, rev_word_map, recent_bleu4, print_freq
    # Training parameters
    start_epoch = 1
    epochs = 120  # number of epochs to train for (if early stopping is not triggered)
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    batch_size = 32
    workers = 1  # for data-loading; right now, only 1 works with h5py
    encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
    decoder_lr = 4e-4  # learning rate for decoder
    grad_clip = 5.  # clip gradients at an absolute value of
    alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
    best_bleu4 = 0.  # BLEU-4 score right now
    recent_bleu4 = 0. # BLUE-4 score at last epoch
    print_freq = 100  # print training/validation stats every __ batches
    fine_tune_encoder = False
    #checkpoint = None  # path to checkpoint, None if none
    checkpoint = data_checkpoint

    """
    Training and validation.
    """
    # Read word map
    with open(data_word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    vocab_size = len(word_map)

    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                    embed_dim=emb_dim,
                                    decoder_dim=decoder_dim,
                                    vocab_size=vocab_size,
                                    dropout=dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=decoder_lr)

    target_decoder = DecoderWithAttention(attention_dim=attention_dim,
                                    embed_dim=emb_dim,
                                    decoder_dim=decoder_dim,
                                    vocab_size=vocab_size,
                                    dropout=dropout)
    target_decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=decoder_lr)

    if args.which_model == "resnet101":
        encoder = ResNet101Encoder()  # I kept it as text right now but you can import model
        print("ResNet101Encoder")
    elif args.which_model == "resnet152":
        encoder = ResNet152Encoder()
        print("ResNet152Encoder")
    elif args.which_model == "resnet50":
        encoder = ResNet50Encoder()
        print("ResNet50Encoder")
    else:
        print(
            f"User selected {args.which_model} model not found.\r\nPlease select one of the available models ('resnet101' or 'resnet152') correctly."
        ) 
        exit()

    if args.target_model == "resnet101":
        target_encoder = ResNet101Encoder()  # I kept it as text right now but you can import model
        print("Target: ResNet101Encoder")
    elif args.target_model == "resnet152":
        target_encoder = ResNet152Encoder()
        print("Target: ResNet152Encoder")
    elif args.target_model == "resnet50":
        target_encoder = ResNet50Encoder()
        print("Target: ResNet50Encoder")
    else:
        print(
            f"User selected {args.target_model} model not found.\r\nPlease select one of the available models ('resnet101' or 'resnet152') correctly."
        ) 
        exit()

    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=encoder_lr) if fine_tune_encoder else None

    target_encoder.fine_tune(fine_tune_encoder)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    target_encoder = target_encoder.to(device)
    target_decoder = target_decoder.to(device)

    # Load model
    print(f"Checkpoint name: {data_best_checkpoint}")
    checkpoint = torch.load(data_best_checkpoint, map_location=(str(device)))
    # decoder = checkpoint['decoder']
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    # decoder = decoder.to(device)
    decoder.eval()
    # encoder = checkpoint['encoder']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.fine_tune(False)
    # encoder = encoder.to(device)
    encoder.eval()

    print(f"Target checkpoint name: {data_target_best_checkpoint}")
    target_checkpoint = torch.load(data_target_best_checkpoint, map_location=(str(device)))
    # decoder = checkpoint['decoder']
    target_decoder.load_state_dict(target_checkpoint['decoder_state_dict'])
    # decoder = decoder.to(device)
    target_decoder.eval()
    # encoder = checkpoint['encoder']
    target_encoder.load_state_dict(target_checkpoint['encoder_state_dict'])
    target_encoder.fine_tune(False)
    # encoder = encoder.to(device)
    target_encoder.eval()

    # Custom dataloaders
    attack_normalize = transforms.Compose([transforms.Normalize(mean=data_test_mean, std=data_test_std)])
    # attack_normalize = None

    attack_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=attack_normalize),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Initialize timer
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        end = torch.cuda.Event(enable_timing=True)  
    else:
        start = time.perf_counter()

    success_rate = 0
    data_time = 0
    epsilon = args.epsilon
    print(f"epsilon: {epsilon}")
    
    if(args.export_original_image == "True"):
        temp_ori_output_folder = os.path.join(data_path, "adversarial_samples")
        original_output_folder = os.path.join(temp_ori_output_folder, "original_"+args.which_data)
        if(not os.path.exists(original_output_folder)):
            os.makedirs(original_output_folder, exist_ok = True)
            print("Directory '%s' created successfully" % original_output_folder)

    if(args.export_perturbed_image == "True"):
        temp_ori_output_folder = os.path.join(data_path, "adversarial_samples")
        perturbed_output_folder = os.path.join(temp_ori_output_folder, "perturbed_"+args.which_model+"_"+args.which_data+"_"+str(args.epsilon).replace(".","_"))
        if(not os.path.exists(perturbed_output_folder)):
            os.makedirs(perturbed_output_folder, exist_ok = True)
            print("Directory '%s' created successfully" % perturbed_output_folder)

    if(args.export_caption == "True"):
        caption_output_folder = os.path.join(data_path, "adversarial_samples/caption")
        caption_filename = os.path.join(caption_output_folder, f"{args.which_model}_{args.which_data}.json")
        if(not os.path.exists(caption_output_folder)):
            os.makedirs(caption_output_folder, exist_ok = True)
            print("Directory '%s' created successfully" % caption_output_folder)

    # For each image (batch_size of 32)
    captions = []
    for i, (images, ori_images, image_sizes, filenames, caps, caplens, allcaps) in enumerate(
            tqdm(attack_loader, desc="ATTACK AND EVALUATING AT BEAM SIZE " + str(args.beam_size))):
    # for i, (images, ori_images, image_sizes, filenames, caps, caplens, allcaps) in enumerate(attack_loader):
        if(args.export_caption == "True"):
            captions = save_caption(filenames, caps, caplens, captions, rev_word_map)
            # print(captions)
        total = i+1

        imgs, ori_imgs, signed_grads = create_adversarial_pattern_from_loader(images, 
                                            ori_images, 
                                            caps, 
                                            caplens, 
                                            encoder, 
                                            encoder_optimizer, 
                                            decoder, 
                                            decoder_optimizer, 
                                            criterion)
        perturbed_imgs = UFGSM_attack(ori_imgs, epsilon, signed_grads)
        if(args.export_original_image == "True"):
            save_images(ori_imgs, image_sizes, filenames, original_output_folder)
        if(args.export_perturbed_image == "True"):
            save_images(perturbed_imgs, image_sizes, filenames, perturbed_output_folder)
        success_rate = success_rate + evaluate(ori_imgs, perturbed_imgs, target_encoder, target_decoder, args.beam_size)

    # Export caption
    if(args.export_caption == "True"):
        # Serializing json
        json_object = json.dumps(captions, indent=4)
        # Writing to sample.json
        with open(caption_filename, "w") as outfile:
            outfile.write(json_object)
            print("write")
            # exit()

    # Initialize timer
    if device.type == 'cuda':
        end.record()
        torch.cuda.synchronize()
        data_time += start.elapsed_time(end)
    else:
        data_time += time.perf_counter() - start
    if device.type == 'cuda':
        data_time = data_time * 0.001 / 60 # Convert time from milliseconds to seconds to minutes

    print(f"Success rate: {(success_rate/total/batch_size)*100:.2f}, total runtime: {data_time:.2f} minutes")
    return 1

def save_caption(filenames, caps, caplens, cap_lists, rev_word_map):
    for i in range(len(filenames)):
        temp = {}
        temp["filename"]=filenames[i]
        words=[rev_word_map[ind] for ind in caps[i].numpy().tolist()]
        words.remove("<start>")
        words.remove("<end>")
        #sublist created with list comprehension
        words = [value for value in words if value != "<pad>"]
        temp["caps"]=words
        temp["caplens"]=int(caplens[i])-2
        cap_lists.append(temp)
    return cap_lists

    
def UFGSM_attack(image, eps, signed_grad): #Untargeted FGSM
        # we can add "create adversarial pattern" function here
        perturbed_image = image + (eps*signed_grad) # Untargeted / Gradient ascent / Move away from minima
        perturbed_image = torch.clamp(perturbed_image , -1, 1) # Might need clipping after addition / subtraction
        return perturbed_image

def save_images(imgs, image_sizes, filenames, filepath):
    for i in range(len(imgs)):
        # img_size = image_sizes[i].numpy().tolist()
        # img_size = tuple(img_size)
        # resized_perturbed_image = F.interpolate(imgs[i].unsqueeze(0), img_size)
        resized_img = imgs[i]
        save_path = os.path.join(filepath,filenames[i])

        torchvision.utils.save_image(resized_img, save_path)

        # img_size = [3, img_size[0], img_size[1]]
        # print(img_size)
        # # exit()
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize(img_size),
        #     transforms.ToTensor()
        # ])
        # print(perturbed_imgs[i].unsqueeze(0).shape)
        # perturbed_img = [transform(x_) for x_ in perturbed_imgs[i]]
        # print(perturbed_img[i].shape)
        # tmp_purturbed_image = resized_perturbed_image.squeeze(0).permute(1,2,0).numpy()
        # plt.imshow((tmp_purturbed_image))
        # plt.savefig(f'{save_path}', bbox_inches='tight', dpi=300)
        # print(save_path)
        # exit()

def evaluate(ori_images, perturbed_imgs, encoder, decoder, beam_size):
    success = 0
    for i in range(len(ori_images)):
        image = ori_images[i].unsqueeze(0)
        perturbed_image = perturbed_imgs[i].unsqueeze(0)
        # signed_gradient = signed_grads[i].unsqueeze(0)

        # imgs[0].shape, perturbed_imgs[0].shape
        # image.shape, perturbed_image.shape, signed_gradient.shape
        # print(image.shape)
        # print(perturbed_image.shape)
        seq, _ = caption_image_beam_search_for_perturbed_image(encoder, decoder, image, word_map, beam_size)
        # words = [rev_word_map[ind] for ind in seq]
        # print(f"Normal {words}")

        adv_seq, _ = caption_image_beam_search_for_perturbed_image(encoder, decoder, perturbed_image, word_map, beam_size)
        # adv_words = [rev_word_map[ind] for ind in adv_seq]
        # print(f"Adv word {adv_words}")
        if(adv_seq != seq):
            # print("Attack Success")
            success = success + 1
        # else:
            # print("Attack Failed")
    return success
    # print(f"success rate: {success/len(ori_images)*100:.2f}")

def create_adversarial_pattern_from_loader(image, ori_image, caps, caplens, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion):
    """
    Performs one epoch's validation.

    :param attack_loader: DataLoader for validation data.
    :return: BLEU-4 score
    """
    decoder.train() # For gradient calculation during validation
    if encoder is not None:
        encoder.eval() # eval mode (no dropout or batchnorm)

    # Move to device, if available
    ori_imgs = ori_image.to(device)
    imgs = image.to(device)
    caps = caps.to(device)
    caplens = caplens.to(device)

    # Prepare for gradient calculation of loss w.r.t to the input image.
    imgs.requires_grad = True # *Adversarial Attack*
    
    # Forward propagation through encoder
    if encoder is not None:
        encoded_imgs = encoder(imgs)
    
    # Forward propagation through decoder
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoded_imgs, caps, caplens)

    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    targets = caps_sorted[:, 1:]

    # Remove timesteps that we didn't decode at, or are pads
    # pack_padded_sequence is an easy trick to do this
    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

    # Calculate loss
    loss = criterion(scores, targets)

    # Add doubly stochastic attention regularization
    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    
    # Clean optimizers' gradients
    decoder_optimizer.zero_grad()
    if encoder_optimizer is not None:
        encoder_optimizer.zero_grad()

    # Back propagation for *Adversarial Attack*
    imgs.retain_grad() # Explicitly allow leaf node (input) calculation
    loss.backward() # Calculates img.grad = d-loss/d-img for every img with img.requires_grad=True
    
    # Result of back propagation
    gradients = imgs.grad

    # Getting gradient sign (+,-) for adversarial perturbation
    signed_grads = torch.sign(gradients)
    
    # Stopping gradient calculation of input and gradient sign
    imgs = imgs.detach()
    signed_grads = signed_grads.detach()

    # Denormalize input images
    # inv_data_test_mean = [1/x for x in data_test_mean]
    # inv_data_test_std = [-x for x in data_test_std]
    # invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
    #                                                     std = inv_data_test_mean),
    #                                 transforms.Normalize(mean = inv_data_test_std,
    #                                                     std = [ 1., 1., 1. ]),
    #                             ])
    # signed_grads = invTrans(signed_grads)

    decoder.eval() # Explicitly say evaluation

    return imgs, ori_imgs, signed_grads

# get caption after adversarial attack
def caption_image_beam_search_for_perturbed_image(encoder, decoder, purterbed_image, word_map, beam_size):
    k = beam_size # beam_size

    vocab_size = len(word_map)

    # Encode, we've already encode the image :)
    encoder_out = encoder(purterbed_image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    # encoder_out = purterbed_image # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k (beam_size)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    # if beam size = 3
    # 1 <start>
    # 2 <start>
    # 3 <start>

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

    smth_wrong = False
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
            smth_wrong = True # Predict wrong too many time and cannot came to <end> conclusion
            break
        step += 1

    if not smth_wrong:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
    else:
        seq = []
        alphas = []
    
    return seq, alphas


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--which_model", default="resnet101", type=str, 
    help="Which model to use 'resnet50', 'resnet101', or 'resnet152'", choices=["resnet50", "resnet101", "resnet152"])
    argparser.add_argument("-t", "--target_model", default="resnet101", type=str, 
    help="Targeting model to be attacked by which_model", choices=["resnet50", "resnet101", "resnet152"])
    argparser.add_argument("-d", "--which_data", default="coco2014", type=str, 
    help="Which dataset to use 'coco2014', or 'flickr8k'", choices=["coco2014", "flickr8k"])
    argparser.add_argument("-b", "--beam_size", default=3, type=int,
    help="Beam size at which to generate captions for evaluation", choices=[1, 2, 3, 4, 5, 6, 7, 8])
    argparser.add_argument("-e", "--epsilon", default=0.004, type=float,
    help="Epsilon at which to create perturbation images", choices=[0.004, 0.02, 0.04, 0.1, 0.2, 0.3, 0.4])
    argparser.add_argument('-o', '--export_original_image', type=str, default="False",
    help='Export compressed original image flag, set to True if you want to export compressed original images.', choices=['True', 'False'])
    argparser.add_argument('-p', '--export_perturbed_image', type=str, default="False",
    help='Export compressed perturbed image flag, set to True if you want to export compressed perturbed images.', choices=['True', 'False'])
    argparser.add_argument('-c', '--export_caption', type=str, default="False",
    help='Export caption flag, set to True if you want to export caption.', choices=['True', 'False'])
    # argparser.add_argument('-cp', '--export_caption_path', type=str, default="/scratch/ps4534/ml/adversarial-attack-to-caption/data/captions/",
    # help='Path to export captions')
    # argparser.add_argument('-ip', '--export_image_path', type=str, default="/scratch/ps4534/ml/adversarial-attack-to-caption/data/images/",
    # help='Path to export images')
    return argparser.parse_args()

if __name__ == "__main__":
    main()