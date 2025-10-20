import os
import json
import pandas as pd
import re
from tqdm import tqdm
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import requests
from io import BytesIO
import math
import random

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def read_data(path):
    if '.jsonl' in path:
        with open(path, 'r') as json_file:
            data_output = [json.loads(line) for line in json_file]
    else:
        with open(path, 'r') as json_file:
            data_output = json.load(json_file)
    return data_output


def eval_model(args, tokenizer, model, image_processor, context_len):
    # Model
    disable_torch_init()

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    if 'trustvl' in args.model_path:
        source_ids = []
        for image_file in image_files:
            source_id = torch.tensor(1, dtype=torch.long)
            source_ids.append(source_id)
        source_ids = torch.stack(source_ids).to(model.device, dtype=torch.long)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                source_ids=source_ids,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
    else:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # print(outputs)
    
    return outputs


def main(args):
    data_eval = read_data(args.question_file)
    questions = get_chunk(data_eval, args.num_chunks, args.chunk_idx)

    if os.path.exists(args.answers_file):
        start_idx = len(read_data(args.answers_file))
    else:
        start_idx = 0
    questions = questions[start_idx:]

    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    # model_base = None # None for fine-tuned models, "lmsys/vicuna-13b-v1.5" for lora
    model_base = args.model_base
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    print(f'****{model_path}, {model_name}, {model_base}')

    if args.enable_direct_evidence:
        print('****Enable Direct Evidence')
    else:
        print('****Disable Direct Evidence')
    if args.enable_inverse_evidence:
        print('****Enable Inverse Evidence')
    else:
        print('****Disable Inverse Evidence')
    

    print(model_path)
    print(model.device)

    path_save = os.path.expanduser(args.answers_file)
    
    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    print(path_save)
    
    for i, row in enumerate(tqdm(questions)):
        new_dict = {}
        
        caption = row['caption']
        image_file = args.image_folder + row['image_path'][1:]
        if args.enable_direct_evidence:
            input_direct_evidence = row['input_direct_evidence']
        else:
            input_direct_evidence = []
        if args.enable_inverse_evidence:
            input_inverse_evidence = row['input_inverse_evidence']
        else:
            input_inverse_evidence = []

                
        if 'trustvl' in model_path:
            input_text = "<text> {text} </text> \n\n<direct evidence> {direct_evidence} </direct evidence> \n\n<inverse evidence> {inverse_evidence} </inverse evidence> \n\nIs there any cross-modal misinformation?"
            input_prompt = input_text.format(text=row['caption'],
                                direct_evidence='; '.join(input_direct_evidence),
                                inverse_evidence='; '.join(input_inverse_evidence))

        else:
            input_text = "<text> {text} </text> \n\n<context evidence> {context_evidence} </context evidence> \n\nDoes the context evidence support or refute the text? You should answer: \n- 'Support' if the text is real and supported by context evidence. \n- 'Refute' if the text refuted by context evidence."

            input_prompt = input_text.format(text=row['caption'],
                                context_evidence='; '.join(input_inverse_evidence))

                    
        eval_args = type('Args', (), {
            "model_path": model_path,
            "model_base": model_base,
            "model_name": get_model_name_from_path(model_path),
            "query": input_prompt,
            "conv_mode": "trustvl_v1",
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 1024
        })()

        answer = eval_model(eval_args, tokenizer, model, image_processor, context_len)
        
        new_dict['question_id'] = row['idx']
        new_dict['news_text'] = row['caption']
        new_dict['image_path'] = row['image_path']
        new_dict['fake_cls']= row['source_id']

        new_dict["turns"] = []
        new_dict["turns"].append(input_prompt)

        choices = []
        choices.append({"index": 0, "turns": [answer]})
        new_dict["choices"]= choices
        new_dict["reference"] = row["reference"]

        with open(path_save, 'a+') as fw_json: 
            fw_json.write(json.dumps(new_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/trustvl-13b-task") 
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="./data/eval/NewsCLIPpings_7264.jsonl")
    parser.add_argument("--image-folder", type=str, default="./data/origin")
    parser.add_argument("--answers-file", type=str, default="./outputs/eval/answer_newsclippings.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--enable-direct-evidence", action=argparse.BooleanOptionalAction)
    parser.add_argument("--enable-inverse-evidence", action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()

    main(args)
