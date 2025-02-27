from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.model_vqa import eval_model
import os
import json
from tqdm import tqdm
import math
import shortuuid
from llava.utils import disable_torch_init
import random
import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Script based on https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):

    print("start")
    # MODEL
    disable_torch_init() 
    MODEL_PATH = "path/to/models/"
    model_id = args.model_id 
    model_path = MODEL_PATH + model_id
    model_name = get_model_name_from_path(model_path) if args.model_name is None else args.model_name
    device_map = "auto"
    overwrite_config = {}
    overwrite_config["mm_patch_merge_type"] = "flat"
    model_base = MODEL_PATH + args.model_base if args.model_base is not None else None
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, device_map=device_map,overwrite_config=overwrite_config) # Add any other thing you want to pass in llava_model_args
   
   # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    outs = []    

    for line in tqdm(questions):
            # READ FILE
            idx = line["id"]
            gt = line["conversations"][1]["value"]
            image_files = line["video-seq"]
            # obtain action_number
            action_path = line["video-seq"][-1][0]  # Take the first element from the last video-seq
            action_number = int(action_path.split("/")[0].split("_")[1])  # Extract and parse action number 
            if action_number == 4:
                action_number = len(line["video-seq"])
            
            qs = line["conversations"][0]["value"]
            cur_prompt = args.extra_prompt + qs

            # CONVERSATION
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt(add_roles=False)

            # INPUT IDS INCLUDING THE IMAGE TOKEN IN THE RIGHT POSITION(S)
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda() # copied this from llava video demo
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # LOAD IMAGE FILE(S), PREPROCESS AND TAKE THE PIXEL VALUES
            image_tensors = []
            modalities = []
            for j, image_frames in enumerate(image_files):
                frame_tensors = []
                modalities_aux = []
                for image_file in image_frames:
                    image = Image.open(os.path.join(args.image_folder, image_file))
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                    frame_tensors.append(image_tensor.half().cuda())
                    modalities_aux.append("video-seq")
                image_tensors.append(frame_tensors)
                modalities.append(modalities_aux)
            # INFERENCE: model.generate is in llava_llama.py and calls prepare_inputs_labels_for_multimodal in llava_arch.py
          
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[image_tensors],
                    attention_mask=attention_masks,
                    modalities = modalities,
                    do_sample= True if args.temperature > 0 else False, 
                    temperature=args.temperature, 
                    top_p=args.top_p, 
                    num_beams=args.num_beams, 
                    max_new_tokens=args.max_new_tokens, 
                    stopping_criteria=[stopping_criteria],
                    use_cache=True,
                    )
                
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()            
            ans_id = shortuuid.uuid()
            outs.append({
                        "dataset": f"action_{action_number}_val", #args.dataset_name,
                        "sample_id": idx,
                        "prompt": cur_prompt,
                        "pred_response": outputs,
                        "gt_response": gt,
                        "shortuuid": ans_id,
                        "model_id": model_name,
                        }) 
            
            

    with open(answers_file, 'w', encoding='utf-8') as f:
         json.dump(outs, f, ensure_ascii=False, indent=4)
    
    print("ANSWERS FILE", args.answers_file)
    print("done!")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--test_size", type=int, default=10000000)
    parser.add_argument("--seed", type=int, default="30")
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    eval_model(args)

    