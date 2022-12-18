import argparse
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models_extended import *
from datasets import *
from utils import *
from params_class import *
from nltk.translate.bleu_score import corpus_bleu

# Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
print(f"Device: {device}")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

def main():
    global data_train_log, data_val_log
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

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, grad_clip, alpha_c, fine_tune_encoder, word_map, recent_bleu4, print_freq
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
    if(args.fine_tune_encoder == "True"):
        fine_tune_encoder = True  # fine-tune encoder? -> Change to True if training with different dataset
    else:
        fine_tune_encoder = False
    #checkpoint = None  # path to checkpoint, None if none

    if args.start_from_scratch == "True":
        checkpoint = None
    else:
        checkpoint = data_checkpoint

    """
    Training and validation.
    """
    # Read word map
    with open(data_word_map_file, 'r') as j:
        word_map = json.load(j)

    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                    embed_dim=emb_dim,
                                    decoder_dim=decoder_dim,
                                    vocab_size=len(word_map),
                                    dropout=dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
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

    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=encoder_lr) if fine_tune_encoder else None
    
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Initialize / load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        # decoder = checkpoint['decoder']
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        # decoder_optimizer = checkpoint['decoder_optimizer']
        # print(checkpoint['decoder_optimizer_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        # encoder = checkpoint['encoder']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        # encoder_optimizer = checkpoint['encoder_optimizer']
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)



    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_normalize = transforms.Normalize(mean=data_train_mean, std=data_train_std)
    val_normalize = transforms.Normalize(mean=data_val_mean, std=data_val_std)
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([train_normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([val_normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            filename = data_best_checkpoint
            save_checkpoint(filename, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4)

        filename = data_checkpoint
        save_checkpoint(filename, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, recent_bleu4)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = 0.  # forward prop. + back prop. time
    data_time = 0. # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    arr_losses = []
    arr_top5accs = []
    arr_index = []
    arr_batch_time = []
    arr_data_time = []
    # start = time.time()
    
    # Initialize timer
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        end = torch.cuda.Event(enable_timing=True)  
    else:
        start = time.perf_counter()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        # data_time.update(time.time() - start)
        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            data_time += start.elapsed_time(end)
        else:
            data_time += time.perf_counter() - start

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        # targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        # batch_time.update(time.time() - start)
        
        # Calculate Batch Time
        if device.type == 'cuda':
            end.record()
            torch.cuda.synchronize()
            batch_time += start.elapsed_time(end) 
        else:
            batch_time += time.perf_counter() - start

        # Print status
        if i % print_freq == 0:
            if device.type == 'cuda':
                batch_time = batch_time * 0.001 # Convert time from milliseconds to seconds
                data_time = data_time * 0.001 # Convert time from milliseconds to seconds
                
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f} secs\t'
                  'Data Load Time {data_time:.3f} secs\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

            arr_losses.append(losses)
            arr_top5accs.append(top5accs)
            arr_index.append(i)
            arr_batch_time.append(batch_time)
            arr_data_time.append(data_time)
            # print(f"{arr_index}\r\n")
            # save_log(data_train_log, "train" , arr_losses, arr_top5accs, None, epoch, arr_index, len(train_loader), arr_batch_time, arr_data_time)
        # Re-Initialize timer
        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            end = torch.cuda.Event(enable_timing=True)  
        else:
            start = time.perf_counter()

    save_log(data_train_log, "train" , arr_losses, arr_top5accs, None, epoch, arr_index, len(train_loader), arr_batch_time, arr_data_time)

def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = 0.
    losses = AverageMeter()
    top5accs = AverageMeter()

    # start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Initialize timer
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        end = torch.cuda.Event(enable_timing=True)  
    else:
        start = time.perf_counter()
    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, ori_imgs, image_sizes, filenames, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            # scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            # targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            # batch_time.update(time.time() - start)


            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            # Calculate Batch Time
            if device.type == 'cuda':
                end.record()
                torch.cuda.synchronize()
                batch_time += start.elapsed_time(end) 
            else:
                batch_time += time.perf_counter() - start

            if i % print_freq == 0:
                if device.type == 'cuda':
                    batch_time = batch_time * 0.001 # Convert time from milliseconds to seconds

                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time:.3f} secs\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Re-Initialize timer
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                end = torch.cuda.Event(enable_timing=True)  
            else:
                start = time.perf_counter()

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

        save_log(data_val_log, "val", losses.avg, top5accs.avg, bleu4)


    return bleu4


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("-m", "--which_model", default="resnet101", type=str, 
    # help="Which model to use 'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    argparser.add_argument("-m", "--which_model", default="resnet101", type=str, 
    help="Which model to use 'resnet50', 'resnet101', or 'resnet152'", choices=["resnet50", "resnet101", "resnet152"])
    argparser.add_argument("-d", "--which_data", default="coco2014", type=str, 
    help="Which dataset to use 'coco2014', or 'flickr8k'", choices=["coco2014", "flickr8k"])
    argparser.add_argument("-s", "--start_from_scratch", type=str, required=True,
    help="Start from scratch? -> assign `True` only if you want to begin training from scratch", choices=["True", "False"])
    argparser.add_argument("-f", "--fine_tune_encoder", type=str, required=True,
    help="Fine-tune encoder? -> assign `True`", choices=["True", "False"])

    return argparser.parse_args()

if __name__ == "__main__":
    main()
