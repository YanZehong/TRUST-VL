# TRUST-VL: An Explainable News Assistant for General Multimodal Misinformation Detection

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/user-attachments/assets/3bb4b16d-efc3-4d46-b839-2e737445d7bd" width="60%" alt="TRUST-VL" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2509.04448"> 
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arXiv&labelColor=grey"/></a> 
  <a href="https://yanzehong.github.io/trust-vl/" target="_blank"><img alt="Homepage"
    src="https://img.shields.io/badge/TRUST--VL-Homepage-7289da?logo=googlegemini&logoColor=white&color=886FBF"/></a>
  <a href="https://huggingface.co/" target="_blank"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TRUST--VL-ffc107?color=FFD21E&logoColor=white"/></a>
  <a href="https://huggingface.co/" target="_blank"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TRUST--Instruct-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE"><img alt="License"
    src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg"/></a>
  <br>
</div>

## News
[2025/09/06] 🔥 TRUST-VL is realsed. Model checkpoints and TRUST-Instruct is coming soon. Checkout the [paper](https://arxiv.org/abs/2509.04448) for more details.

## Contents
- [Quickstart](#quickstart)
- [Training](#training)
- [Evals](#evals)

## Quickstart
Take your first steps with the TRUST-VL model.

1. Clone this repository and install package
```bash
git clone https://github.com/YanZehong/TRUST-VL.git
cd TRUST-VL
conda create -n trustvl python=3.10 -y
conda activate trustvl
pip install --upgrade pip
pip install -e .

```

<details>
<summary>(Optional) Install additional packages for training cases</summary>

```bash
pip install -e ".[train]"
pip install flash-attn==2.6.3 --no-build-isolation #--no-cache-dir
```

</details>


## Model Weights

Please check out [🤗 Huggingface Models](https://huggingface.co/) for public LLaVA checkpoints.

```
git lfs install
git clone https://huggingface.co
```

## Training
TRUST-VL training consists of three stages:
In Stage 1, we begin by training the projection module for one epoch on 1.2 million image–text pairs (653K news samples from VisualNews and 558K samples from the LLaVA training corpus). This stage aligns the visual features with the language model. In Stage 2, we jointly train  the LLM and the projection module for one epoch using 665K synthetic conversation samples from the LLaVA training corpus to improve the  model’s ability to follow complex instructions. In Stage 3, we fine-tune the full model on 198K reasoning samples from TRUST-Instruct for three epochs to further enhance its misinformation-specific reasoning capabilities.  

Similar to LLaVA, TRUST-VL is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Stage 1: Language-Image Alignment + News Domain Alignment

Please download the 1211K subset we use in the paper [here](), which is based on the [LAION-CC-SBU dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Training script with DeepSpeed ZeRO-2: [`trust_vl_stage1.sh`](https://github.com/YanZehong/TRUST-VL/blob/main/scripts/trust_vl_stage1.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.


### Stage 2: Visual Instruction Tuning

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

Training script with DeepSpeed ZeRO-3: [`trust_vl_stage2.sh`](https://github.com/YanZehong/TRUST-VL/blob/main/scripts/trust_vl_stage2.sh).


### Stage 3: Misinformation Tuning

Please download the annotation of the final mixture our instruction tuning data [TRUST-Instruct_task198k.json](), and download the images from constituting datasets. After downloading all of them, organize the data as follows in `./data`,

```
├── origin
│   ├── bbc
│   ├── guardian
│   ├── usa_today
│   ├── washington_post
│   └── data.json
├── DGM4
│   ├── manipulation
│   ├── metadata
│   └── origin
├── Factify2
│   ├── data
│   └── images-train
├── MMFakeBench
│   ├── fake
│   ├── real
    └── source
```

Training script with DeepSpeed ZeRO-3: [`trust_vl_stage3.sh`](https://github.com/YanZehong/TRUST-VL/blob/main/scripts/trust_vl_stage3.sh).

## Evals
In TRUST-VL, we evaluate models on a diverse set of 7 misinformation benchmarks. For example,

```bash
CUDA_VISIBLE_DEVICES=0 bash /home/zehong/TRUST-VL/scripts/eval/newsclippings.sh
```


## Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)

```bibtex
@article{yan2025trustvl,
  title={{TRUST-VL}: An Explainable News Assistant for General Multimodal Misinformation Detection},
  author={Yan, Zehong and Qi, Peng and Hsu, Wynne and Lee, Mong Li},
  journal={arXiv preprint arXiv:2509.04448},
  year={2025}
}
```

## Acknowledgement
LLaVA, Vicuna: Thanks to their amazing works.

**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the OpenAI Terms of Use for the dataset and the specific licenses for base language models. This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.