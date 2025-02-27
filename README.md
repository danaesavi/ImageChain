# ImageChain
This repository is associated with the research paper [ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2502.19409)

## StoryFrames Dataset
The StoryFrames dataset is [available on Hugging Face](https://huggingface.co/datasets/ingoziegler/StoryFrames).

StoryFrames is a human-annotated dataset created to enhance a model's capability of understanding and reasoning over sequences of images. It is specifically designed for tasks like generating a description for the next scene in a story based on previous visual and textual information. The dataset repurposes the StoryBench dataset, a video dataset originally designed to predict future frames of a video. StoryFrames subsamples frames from those videos and pairs them with annotations for the task of next-description prediction. Each "story" is a sample of the dataset and can vary in length and complexity.

You can load it as follows:

```python
from datasets import load_dataset

ds = load_dataset("ingoziegler/StoryFrames")

# to work with stories containing 3 scenes
ds_3 = ds.filter(lambda sample: sample["num_scenes"] == 3)
```

The dataset is described in detail [here](https://huggingface.co/datasets/ingoziegler/StoryFrames#what-is-a-story-in-storyframes) and features a specific feature description [here](https://huggingface.co/datasets/ingoziegler/StoryFrames#detailed-field-descriptions).

## Code

This repo contains modified scripts of [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).  

**Installation**  

1. **Set up LLaVA-NeXT**  
   Follow the official [LLaVA-NeXT installation guide](https://github.com/LLaVA-VL/LLaVA-NeXT).  

2. **Replace Files**  
   Clone this repo and copy the modified files into your LLaVA-NeXT directory:  
   ```bash
   git clone https://github.com/danaesavi/ImageChain.git
   cp -r ImageChain/src/llava/* LLaVA-NeXT/llava/
   ```
   Modified Files
   ```bash
   └── llava
    ├── constants.py
    ├── conversation.py
    ├── mm_utils.py
    ├── model/
    │   ├── builder.py
    │   ├── language_model/llava_llama.py
    │   └── llava_arch.py
    └── train/train.py
    ```

## Citation
If you use this work, please cite:

**ImageChain**  
```
@misc{villegas2025imagechainadvancingsequentialimagetotext,
      title={ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models}, 
      author={Danae Sánchez Villegas and Ingo Ziegler and Desmond Elliott},
      year={2025},
      eprint={2502.19409},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.19409}, 
}
```

**LLaVA-NeXT**  
```
@misc{liu2024llavanext,
    title={LLaVA-NeXT: Improved reasoning, OCR, and world knowledge},
    url={https://llava-vl.github.io/blog/2024-01-30-llava-next/},
    author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
    month={January},
    year={2024}
}
```
