import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import argparse
import time
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
from params_class import *

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device('cpu')

#@title Model

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

#@title Caption prediction

def generate_beam(model, tokenizer, beam_size: int = 3, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                # next_tokens_source = next_tokens // scores_sum.shape[1]
                next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def main():
    # Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    print(f"Device: {device}")

    # Initialize timer
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        end = torch.cuda.Event(enable_timing=True)  
    else:
        start = time.perf_counter()

    success = 0
    data_time = 0
    success, total_image_found = evaluate(device)

    # Initialize timer
    if device.type == 'cuda':
        end.record()
        torch.cuda.synchronize()
        data_time += start.elapsed_time(end)
    else:
        data_time += time.perf_counter() - start
    if device.type == 'cuda':
        data_time = data_time * 0.001 / 60 # Convert time from milliseconds to seconds to minutes

    print(f"Success rate: {(success/total_image_found)*100:.2f} ({success}/{total_image_found}), total runtime: {data_time:.2f} minutes")
    return 1



def evaluate(device):
    
    #@title CLIP model + GPT2 tokenizer
    args = _parse_arguments()

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #@title Load model weights

    prefix_length = 10 # As same as in the pre-trained model weight

    clip_cap_model = ClipCaptionModel(prefix_length)

    if(args.which_data == "coco2014"):
        model_path = os.path.join(data_path, "checkpoints/coco_weights.pt")
    elif(args.which_data == "flickr8k"):
        model_path = os.path.join(data_path, "checkpoints/conceptual_weights.pt")
    else:
        print("Data not found.")
        exit()

    clip_cap_model.load_state_dict(torch.load(model_path, map_location=CPU)) 

    clip_cap_model = clip_cap_model.eval() 
    clip_cap_model = clip_cap_model.to(device)

    #@title Inference
    use_beam_search = args.use_beam_search #@param {type:"boolean"}  
    beam_size = args.beam_size
    epsilon = args.epsilon
    # original_image_path = args.original_image_path
    # perturbed_image_path = args.perturbed_image_path
    original_image_path = os.path.join(data_path, "adversarial_samples/original_"+args.which_data)
    perturbed_image_path = os.path.join(data_path, "adversarial_samples/perturbed_"+args.which_model+"_"+args.which_data+"_"+str(epsilon).replace(".", "_"))

    attack_image_list = [x for x in list(os.listdir(perturbed_image_path))]
    attack_image_list.sort()
    # print(attack_image_list)
    # exit()
    # assign directory

    # iterate over files in
    # that directory
    success = 0
    total_image_found = 0
    eps_desc = f", eps = {epsilon}"

    if(use_beam_search == "True"):
        beam_desc = f", beam size = {beam_size}"
    else:
        beam_desc = ""


    for filename in tqdm(attack_image_list, desc=f"EVALUATING CLIPcap model {eps_desc + beam_desc} "):
        ori_f = os.path.join(original_image_path, filename)
        adv_f = os.path.join(perturbed_image_path, filename)
        # checking if it is a file
        if os.path.isfile(ori_f) and os.path.isfile(adv_f):
            total_image_found = total_image_found + 1
            ori_image = io.imread(ori_f)
            pil_ori_image = PIL.Image.fromarray(ori_image)
            adv_image = io.imread(adv_f)
            pil_adv_image = PIL.Image.fromarray(adv_image)
        
            # pil_img = Image(filename=UPLOADED_FILE)
            # display(pil_image)
            ori_text_prefix = generate_caption(pil_ori_image, clip_model, preprocess, clip_cap_model, tokenizer, beam_size, prefix_length, use_beam_search, device)
            adv_text_prefix = generate_caption(pil_adv_image, clip_model, preprocess, clip_cap_model, tokenizer, beam_size, prefix_length, use_beam_search, device)
            # print(ori_text_prefix)
            # print(adv_text_prefix)
            # print()

        if(ori_text_prefix != adv_text_prefix):
            # print("Attack Success")
            success = success + 1
    return success, total_image_found

def generate_caption(pil_image, clip_model, preprocess, clip_cap_model, tokenizer, beam_size, prefix_length, use_beam_search, device):
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = clip_cap_model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search == "True":
        generated_text_prefix = generate_beam(clip_cap_model, tokenizer, beam_size=beam_size, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(clip_cap_model, tokenizer, embed=prefix_embed)
    return generated_text_prefix

def _parse_arguments():
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('-o', '--original_image_path', type=str, default="/scratch/ps4534/ml/adversarial-attack-to-caption/data/images/original/",
    # help='Path to input perturbed images')
    # argparser.add_argument('-p', '--perturbed_image_path', type=str, default="/scratch/ps4534/ml/adversarial-attack-to-caption/data/images/perturbed_resnet152_0_004/",
    # help='Path to input perturbed images')
    # argparser.add_argument('-e', '--epsilon_placeholder', type=str, default="0.004 (1/255)",
    # help='Indicate the epsilon_placehoder (if you would like to)')
    argparser.add_argument("-m", "--which_model", type=str,
    help="Which model to use 'resnet50', 'resnet101', or 'resnet152'", choices=["resnet50", "resnet101", "resnet152"])
    argparser.add_argument("-d", "--which_data", type=str,
    help="Which dataset to use 'coco2014', or 'flickr8k'", choices=["coco2014", "flickr8k"])
    argparser.add_argument("-e", "--epsilon", type=float,
    help="Epsilon according to the adversarial samples, you've created by using attack_args.py", choices=[0.004, 0.02, 0.04, 0.1, 0.2, 0.3, 0.4])
    argparser.add_argument("-b", "--use_beam_search", default="False", type=str,
    help="Use beam search flag, set True if you wants to use beam search", choices=['True', 'False'])
    argparser.add_argument("-s", "--beam_size", default=3, type=int,
    help="Beam size at which to generate captions for evaluation", choices=[ 1, 2, 3, 4, 5, 6, 7, 8])
    return argparser.parse_args()

if __name__ == "__main__":
    main()