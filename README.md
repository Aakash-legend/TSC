# Tree Species Classifier (TSC)

A deep learning application that classifies tree species from images using a fine-tuned ResNet50V2 model.

## Features
- Classifies 30 different tree species
- Web interface built with Streamlit
- Transfer learning with ResNet50V2

## Supported Species
amla, asopalav, babul, bamboo, banyan, bili, cactus, champa, coconut, garmalo, gulmohor, gunda, jamun, kanchan, kesudo, khajur, mango, motichanoti, neem, nilgiri, other, pilikaren, pipal, saptaparni, shirish, simlo, sitafal, sonmahor, sugarcane, vad

## Installation

```bash
pip install streamlit tensorflow pillow numpy
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then upload an image of a tree to get the classification result.

## Model Details
- **Architecture**: ResNet50V2 with custom classification head
- **Input Size**: 224x224 RGB images
- **Training**: Transfer learning with 2-phase fine-tuning
- **Dataset**: Tree Species Identification Dataset (30 classes)

## Files
- `app.py` - Streamlit web application
- `TSC.ipynb` - Training notebook
- `resnet50v2_finetuned_30class_v2.keras` - Final trained model
- `Tree_Species_Dataset/` - Training images (30 class folders)