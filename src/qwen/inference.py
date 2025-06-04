'''
This script is used to evaluate the ImageChain model based on Qwen2-VL, on a dataset of video sequences, extracting images and generating responses based on a chat transcript format.
'''

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os 
import math
import torch
import json
import random
import numpy as np
from tqdm import tqdm
import argparse
import warnings
import shortuuid
import re
warnings.filterwarnings("ignore", category=UserWarning)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_chat_transcript_to_messages(qs, image_paths):
    messages = []
    idx = 0

    # Extract the system prompt: everything before the first "USER:"
    user_start = qs.find("USER:")
    if user_start == -1:
        raise ValueError("No USER: found in input.")

    system_prompt = qs[:user_start].strip()
    rest = qs[user_start:].strip()

    # Use a custom system prompt or fallback to extracted one
    sp = ("You are a helpful assistant. In the following, you will be presented with a sequence of images, "
          "each accompanied by its own description. The order of these images is crucial, as each image builds "
          "upon or relates to its predecessors in the sequence. Your goal is to produce the most accurate description "
          "for each image, taking into account its place in the sequence and its relationship to previous images.")
    
    messages.append({
        "role": "system",
        "content": sp #or system_prompt,
    })

    # Split into turns using <s> (removes any surrounding whitespace)
    turns = re.split(r'\s*<s>\s*', rest)

    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
        assistant_start = turn.find("ASSISTANT:")
        user_turn = turn[:assistant_start].strip()
        assistant_turn = turn[assistant_start:].strip()
        
        # Check if it's a user turn
        if user_turn.startswith("USER:"):
            #print("USER:")
            user_text = user_turn[len("USER:"):].strip()
            parts = []
            img_pattern = r'<Image><image></Image>'
            tokens = re.split(f'({img_pattern})', user_text)
            for token in tokens:
                if token == '<Image><image></Image>':
                    if idx >= len(image_paths):
                        raise IndexError("Not enough image paths for placeholders.")
                    parts.append({"type": "image", "image": image_paths[idx]})
                    idx += 1
                elif token.strip():
                    parts.append({"type": "text", "text": token.replace("</s>","").strip()})
            messages.append({
                "role": "user",
                "content": parts
            })

        if assistant_turn.startswith("ASSISTANT:"):
            #print("ASSISTANT:")
            assistant_text = assistant_turn[len("ASSISTANT:"):].strip()
            messages.append({
                "role": "assistant",
                "content": assistant_text.replace("</s>","").strip()
            })

    return messages[:-1]

def eval_model(args):
    """
    Evaluate the Qwen2-VL model on a dataset of video sequences, generating responses based on a chat transcript format.
    Args:
        args (argparse.Namespace): Command line arguments containing model paths, dataset paths, and other configurations.
    """

    print("start")
    # Model and processor
    model_name = "qwen2_vl-7B"
    model_path =  args.model_path 
    instruct_path = args.instruct_path
    adapter_path = args.adapter
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if adapter_path != "":
        print("Adapter:",adapter_path)
        model.load_adapter(adapter_path)

    processor = AutoProcessor.from_pretrained(model_path)
    processor_instruct = AutoProcessor.from_pretrained(instruct_path)

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if args.testing:
        questions = questions[:20]
    outs = [] 
    for line in tqdm(questions):
        # READ FILE
        idx = line["id"]
        print("ID", idx)
        gt = line["conversations"][1]["value"]
        image_files = line["video-seq"]
        # obtain action_number
        action_number = len(line["video-seq"])
        if action_number < 7:
            print("action_number", action_number)
            # IMAGES
            image_paths = []
            for frames in image_files:
                im = os.path.join(args.image_folder, frames[0]) # choosing the first frame in each scene
                image_paths.append(im) 

            qs = line["conversations"][0]["value"]
            messages = parse_chat_transcript_to_messages(qs,image_paths)
   
            # Preparation for inference
            text = processor_instruct.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,temperature = args.temperature, top_p = args.top_p)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            ans_id = shortuuid.uuid()
            outs.append({
                            "dataset": f"action_{action_number}_val", 
                            "sample_id": idx,
                            "prompt": text,
                            "pred_response": output_text,
                            "gt_response": gt,
                            "shortuuid": ans_id,
                            "model_id": model_name,
                            }) 
    with open(answers_file, 'w', encoding='utf-8') as f:
         json.dump(outs, f, ensure_ascii=False, indent=4)
    
    print("ANSWERS FILE", args.answers_file)
    print("done!")



def upload_model_to_hub(args, private=True):
    """
    Load a model from a local path (and optionally an adapter), then upload to Hugging Face Hub.

    Args:
        repo_id (str): Hugging Face repo id e.g. 'username/repo_name'.
        adapter_path (str): Optional path to LoRA adapter to merge.
        private (bool): If True, create repo as private.
    """
    print("start")
    repo_id = args.repo_id
    model_path =  args.model_path 
    adapter_path = args.adapter

    print(f"Loading model from {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if adapter_path != "":
        print("Adapter:",adapter_path)
        model.load_adapter(adapter_path)

    # default processor
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"Pushing to Hugging Face Hub at {repo_id}...")
    # Save model and processor
    model.push_to_hub(repo_id, private=private)
    processor.push_to_hub(repo_id, private=private)
    print(f"Successfully uploaded {repo_id} to Hugging Face Hub!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B")
    parser.add_argument("--instruct_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--repo_id", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default="30")
    parser.add_argument("--testing", action="store_true", help="testing")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.1)

    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    eval_model(args)
    

