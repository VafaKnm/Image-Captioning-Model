# Image Captioning Model

A deep learning project for generating natural language captions for images using the **Flickr8k** dataset.

This project was originally developed in a Kaggle Notebook environment and demonstrates a classic image captioning pipeline using:

* VGG16 for image feature extraction
* Tokenization and sequence preprocessing for captions
* Embedding and LSTM layers for language generation
* A combined vision-language neural network for caption prediction

---

## Overview

Image captioning is a computer vision and natural language processing task where a model receives an image and generates a textual description.

Example:

```text
Input:
    an image of a dog running on grass

Output:
    "a dog is running through the grass"
```

This task combines two different AI domains:

| Domain                      | Role                           |
| --------------------------- | ------------------------------ |
| Computer Vision             | Understand the image content   |
| Natural Language Processing | Generate a meaningful sentence |

---

## What This Project Does

This project shows how to:

* Load the Flickr8k image captioning dataset
* Extract image features using a pretrained VGG16 model
* Clean and preprocess image captions
* Add start and end tokens to captions
* Tokenize captions into integer sequences
* Build a sequence generation model
* Train an image captioning model
* Generate captions for unseen images

---

## Dataset

The project uses the **Flickr8k** dataset.

Flickr8k is a popular image captioning dataset that contains:

* Around 8,000 images
* Multiple human-written captions for each image
* Images covering people, animals, outdoor scenes, activities, and daily-life objects

In the notebook, images are loaded from:

```text
../input/flickr8k/Images
```

Captions are loaded from:

```text
../input/flickr8k/captions.txt
```

---

## Dataset Structure

A typical Flickr8k dataset structure looks like this:

```text
flickr8k/
│
├── Images/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
│
└── captions.txt
```

The captions file maps each image to one or more textual descriptions.

Example:

```text
image.jpg,a dog is running in the grass
image.jpg,a brown dog plays outside
```

---

## Image Feature Extraction

The project uses a pretrained **VGG16** model.

VGG16 is used as a feature extractor instead of training a CNN from scratch.

The notebook removes the final classification layer and uses the second-to-last dense layer as the image representation.

The extracted image feature vector has the shape:

```text
4096
```

Conceptually:

```text
Input Image
    ↓
VGG16
    ↓
Feature Vector
    ↓
4096-dimensional image representation
```

---

## Why Use VGG16?

VGG16 is a pretrained convolutional neural network trained on ImageNet.

It can recognize general visual patterns such as:

* Objects
* Shapes
* Textures
* Scenes
* Visual structures

Using VGG16 saves training time because the model does not need to learn visual features from scratch.

In this project, VGG16 acts as the image encoder.

---

## Caption Preprocessing

Captions are cleaned and converted into a format suitable for sequence generation.

The notebook performs steps such as:

* Convert text to lowercase
* Remove unnecessary characters
* Remove very short tokens
* Add a start token
* Add an end token

Each caption is converted into this format:

```text
startseq a dog is running in the grass endseq
```

The tokens are important:

| Token      | Purpose                                              |
| ---------- | ---------------------------------------------------- |
| `startseq` | Tells the model where caption generation begins      |
| `endseq`   | Tells the model where caption generation should stop |

---

## Tokenization

The cleaned captions are tokenized using a Keras `Tokenizer`.

The notebook reports:

```text
vocab size: 8485
max len of captions: 35
```

Where:

| Value  | Meaning                                    |
| ------ | ------------------------------------------ |
| `8485` | Number of unique tokens in the vocabulary  |
| `35`   | Maximum caption length after preprocessing |

Captions are converted into integer sequences before training.

Example:

```text
startseq a dog runs endseq
```

may become:

```text
[1, 3, 18, 245, 2]
```

---

## Model Architecture

The project uses a two-input neural network.

### Input 1: Image Features

The image branch receives the VGG16 feature vector.

```text
Image → VGG16 → 4096-dimensional feature vector
```

### Input 2: Partial Caption Sequence

The text branch receives the current partial caption sequence.

```text
startseq a dog
```

The model learns to predict the next word.

### Combined Model

Conceptually:

```text
Image Feature Vector ── Dense Layer ┐
                                    ├── Merge/Add ── Dense Layer ── Next Word Prediction
Caption Sequence ── Embedding ── LSTM ┘
```

This architecture allows the model to generate captions word by word while conditioning on the image features.

---

## Training Logic

The model is trained as a next-word prediction system.

For each caption, the model creates multiple training samples.

Example caption:

```text
startseq a dog runs in grass endseq
```

Training pairs:

```text
Input: image_features + "startseq"
Output: "a"

Input: image_features + "startseq a"
Output: "dog"

Input: image_features + "startseq a dog"
Output: "runs"

Input: image_features + "startseq a dog runs"
Output: "in"
```

This teaches the model how to generate captions step by step.

---

## General Pipeline

The full pipeline can be described as:

```text
Flickr8k Images
      ↓
VGG16 Feature Extraction
      ↓
4096-dimensional Image Features
      ↓
Caption Cleaning
      ↓
Tokenizer
      ↓
Sequence Padding
      ↓
Image Feature + Caption Sequence
      ↓
Embedding + LSTM Decoder
      ↓
Next Word Prediction
      ↓
Generated Caption
```

---

## Input and Output

### Input

The input is an image.

Example:

```text
sample_image.jpg
```

### Output

The output is a generated caption.

Example:

```text
a dog is running through the grass
```

---

## Example Caption Generation

A typical caption generation function works like this:

```python
def generate_caption(model, tokenizer, image_feature, max_length):
    caption = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        prediction = model.predict([image_feature, sequence], verbose=0)
        predicted_word_id = prediction.argmax()

        predicted_word = word_for_id(predicted_word_id, tokenizer)

        if predicted_word is None:
            break

        caption += " " + predicted_word

        if predicted_word == "endseq":
            break

    caption = caption.replace("startseq", "").replace("endseq", "").strip()
    return caption
```

---

## Repository Structure

Current repository structure:

```text
Image-Captioning-Model/
│
├── README.md
└── image-captioning-on-flickr8k-dataset.ipynb
```

Suggested future structure:

```text
Image-Captioning-Model/
│
├── README.md
├── requirements.txt
├── notebooks/
│   └── image-captioning-on-flickr8k-dataset.ipynb
├── src/
│   ├── feature_extraction.py
│   ├── caption_preprocessing.py
│   ├── data_generator.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── models/
│   ├── caption_model.h5
│   ├── tokenizer.pkl
│   └── image_features.pkl
├── assets/
│   ├── sample_image.jpg
│   ├── sample_caption_result.png
│   └── model_architecture.png
└── examples/
    └── sample_predictions.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/VafaKnm/Image-Captioning-Model.git
cd Image-Captioning-Model
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install common dependencies:

```bash
pip install numpy pandas matplotlib pillow tensorflow keras scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
image-captioning-on-flickr8k-dataset.ipynb
```

---

## Suggested `requirements.txt`

```txt
numpy
pandas
matplotlib
pillow
tensorflow
keras
scikit-learn
jupyter
```

---

## Example Feature Extraction Code

A reusable VGG16 feature extraction function could look like this:

```python
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np

def build_vgg16_feature_extractor():
    base_model = VGG16()
    model = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output
    )
    return model

def extract_image_feature(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)
    return feature
```

---

## Example Caption Cleaning Function

```python
def clean_caption(caption):
    caption = caption.lower()
    caption = caption.replace("[^a-z]", " ")
    caption = caption.replace("\\s+", " ")
    caption = " ".join([
        word for word in caption.split()
        if len(word) > 1
    ])
    caption = "startseq " + caption + " endseq"
    return caption
```

---

## Evaluation

The current README does not report final numerical evaluation metrics.

For image captioning, common evaluation metrics include:

| Metric  | Description                                           |
| ------- | ----------------------------------------------------- |
| BLEU    | Measures n-gram overlap with reference captions       |
| METEOR  | Considers precision, recall, and synonym matching     |
| ROUGE-L | Measures longest common subsequence                   |
| CIDEr   | Designed specifically for image captioning evaluation |
| SPICE   | Measures semantic scene graph similarity              |

A stronger version of this project should include at least BLEU scores.

Example:

```python
from nltk.translate.bleu_score import corpus_bleu

actual, predicted = [], []

# actual: list of reference captions
# predicted: list of generated captions

score = corpus_bleu(actual, predicted)
print("BLEU:", score)
```

---

## Suggested Result Table

After evaluation, the README can be updated with:

| Metric | Score |
| ------ | ----: |
| BLEU-1 |   TBD |
| BLEU-2 |   TBD |
| BLEU-3 |   TBD |
| BLEU-4 |   TBD |
| METEOR |   TBD |
| CIDEr  |   TBD |

---

## Suggested Improvements

### 1. Add Sample Predictions

The README should include examples like:

| Image        | Generated Caption                    |
| ------------ | ------------------------------------ |
| sample image | `a dog is running through the grass` |
| sample image | `a man is riding a bicycle`          |

This makes the project much easier to understand.

---

### 2. Add BLEU Evaluation

The project should evaluate generated captions using BLEU scores.

Recommended metrics:

```text
BLEU-1
BLEU-2
BLEU-3
BLEU-4
```

---

### 3. Save Model and Tokenizer

Save the trained captioning model:

```python
model.save("models/caption_model.h5")
```

Save the tokenizer:

```python
import pickle

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

---

### 4. Add Inference Script

A clean inference script would make the project usable outside the notebook.

Example command:

```bash
python src/inference.py --image_path assets/sample_image.jpg
```

Expected output:

```text
Generated caption:
a dog is running through the grass
```

---

### 5. Use a Stronger Image Encoder

VGG16 is a good educational baseline, but stronger image encoders can improve results.

Recommended alternatives:

| Encoder            | Notes                               |
| ------------------ | ----------------------------------- |
| ResNet50           | Strong CNN baseline                 |
| InceptionV3        | Common in image captioning projects |
| EfficientNet       | Better accuracy/efficiency          |
| Vision Transformer | Modern image representation         |
| CLIP image encoder | Strong image-text representation    |

---

### 6. Use Attention Mechanism

A more advanced image captioning model can use attention.

Attention helps the decoder focus on different parts of the image while generating each word.

Example:

```text
word "dog"  → focus on dog region
word "grass" → focus on background region
word "running" → focus on pose/action
```

---

### 7. Try Transformer-Based Captioning

Modern image captioning systems often use Transformer architectures.

Recommended future models:

```text
Vision Encoder-Decoder
ViT + GPT-style decoder
BLIP
BLIP-2
OFA
GIT
LLaVA-style multimodal models
```

These models can generate more fluent and accurate captions than a basic CNN-LSTM pipeline.

---

### 8. Add Beam Search

Greedy decoding may choose the most likely word at each step, but it can produce weaker captions.

Beam search keeps multiple candidate captions and often improves generation quality.

Example:

```text
beam width = 3
beam width = 5
```

---

### 9. Add a Demo App

A small demo would make the project easier to test.

Recommended tools:

* Streamlit
* Gradio
* FastAPI

Example UI:

```text
Upload image
      ↓
Extract image features
      ↓
Generate caption
      ↓
Show image + caption
```

---

## Limitations

The current project has some limitations:

* It is notebook-based and not yet structured as reusable Python code.
* The README does not currently include BLEU or other captioning metrics.
* The trained model checkpoint is not included in the repository.
* The tokenizer and extracted image features are not included.
* VGG16 is an older image encoder.
* A basic LSTM decoder may produce repetitive or generic captions.
* Flickr8k is relatively small compared with modern captioning datasets.
* The model may struggle with complex scenes, rare objects, counting, and spatial relationships.

---

## Use Cases

This project can be useful for:

* Learning image captioning
* Understanding CNN-LSTM architectures
* Practicing multimodal deep learning
* Learning feature extraction with pretrained CNNs
* Studying sequence generation
* Building a simple image-to-text demo
* Creating a baseline before using attention or Transformer-based captioning models

---

## Possible Future Work

Recommended future work:

* Add `requirements.txt`
* Add clean Python scripts
* Add saved model file
* Add saved tokenizer
* Add sample input/output images
* Add BLEU evaluation
* Add beam search decoding
* Add attention mechanism
* Add ResNet50 or InceptionV3 encoder
* Add Transformer-based captioning baseline
* Add Streamlit or Gradio demo
* Add Dockerfile
* Add inference API

---

## Conclusion

This repository demonstrates a classic image captioning pipeline using Flickr8k, VGG16 feature extraction, caption preprocessing, tokenization, embeddings, and an LSTM-based decoder.

The project is useful for learning:

```text
image feature extraction
caption preprocessing
sequence tokenization
next-word prediction
CNN-LSTM image captioning
multimodal deep learning
Kaggle-based image captioning workflow
```

With evaluation metrics, sample outputs, saved model artifacts, beam search, and attention-based improvements, this repository can become a much stronger image captioning portfolio project.
