# ImageChain
This repository is associated with the research paper titled ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models

## StoryFrames Dataset
The StoryFrames dataset is ![available on Hugging Face](https://huggingface.co/datasets/ingoziegler/StoryFrames).

StoryFrames is a human-annotated dataset created to enhance a model's capability of understanding and reasoning over sequences of images. It is specifically designed for tasks like generating a description for the next scene in a story based on previous visual and textual information. The dataset repurposes the StoryBench dataset, a video dataset originally designed to predict future frames of a video. StoryFrames subsamples frames from those videos and pairs them with annotations for the task of next-description prediction. Each "story" is a sample of the dataset and can vary in length and complexity.

You can load it as follows:

```python
from datasets import load_dataset

ds = load_dataset("ingoziegler/StoryFrames")

# to work with stories containing 3 scenes
ds_3 = ds.filter(lambda sample: sample["num_scenes"] == 3)
```

The dataset is described in detail ![here](https://huggingface.co/datasets/ingoziegler/StoryFrames#what-is-a-story-in-storyframes) and features a specific feature description ![here](https://huggingface.co/datasets/ingoziegler/StoryFrames#detailed-field-descriptions).
