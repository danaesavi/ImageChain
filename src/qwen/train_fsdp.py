'''
Train a Qwen2-VL model with FSDP using LoRA for the next-scene description task.
This script prepares a dataset from a JSON file, processes images, and trains the model using the SFTTrainer from the trl library.
'''
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
import torch, os, json
from tqdm import tqdm
from qwen_vl_utils import parse_chat_transcript_to_messages, get_chunk, image_transform
from peft import get_peft_model, LoraConfig
from transformers import AutoProcessor
import argparse, random, numpy as np
import wandb
from trl import SFTConfig, SFTTrainer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def prepare_dataset(args):
    # Load data from the specified JSON file and prepare it for training
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.testing:
        questions = questions[:100]
    outs = [] 
    for line in tqdm(questions):
        # READ FILE
        idx = line["id"]
        # parse file 
        gt = line["conversations"][1]["value"].replace("</s>","").strip()
        image_files = line["video-seq"]
        # obtain action_number
        action_number = len(line["video-seq"])        
        print("action_number", action_number)
        # IMAGES
        image_paths = []
        for frames in image_files:
            if "train" not in frames[0]:
                action_n = "4_7" if action_number>3 else action_number
                im = os.path.join(args.image_folder + "action_{}/train/{}".format(action_n,idx),frames[0]) # choosing the first frame in each
            else:
                im = os.path.join(args.image_folder, frames[0]) 
            image_paths.append(im) 

        qs = line["conversations"][0]["value"]
        messages = parse_chat_transcript_to_messages(qs,image_paths)
        messages[-1]["content"] = gt
        outs.append(messages)
    return outs

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor_instruct.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="lora_model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B")
    parser.add_argument("--instruct_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()
    accelerator.print(f"Accelerator device: {accelerator.device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model paths
    
    model_path =  args.model_path 
    instruct_path = args.instruct_path

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_path)
    processor_instruct = AutoProcessor.from_pretrained(instruct_path)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    # Print trainable parameters
    print(model.print_trainable_parameters())
    dataset = prepare_dataset(args)

    # Training args (trl)
    training_args = SFTConfig(
        output_dir= args.output_dir + args.model_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        warmup_ratio=0.03,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="wandb",
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
   
    wandb.init(project="qwen2-7b-imageChain", name=args.model_name, config=training_args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )

    trainer.train()
    peft_model = trainer.model

    # Unwrap from FSDP if needed (depends on wrapping)
    if hasattr(peft_model, "module"):
        peft_model = peft_model.module

    peft_model.save_pretrained(training_args.output_dir)
    
    


    
