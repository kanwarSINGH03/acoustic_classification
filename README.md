# 🧠 Acoustic Roof Sound Classification

This project classifies underground mine roof sounds (e.g., **drummy** vs **tight**) using MFCC-based features extracted from real impact recordings. The model uses an MLP trained on labeled audio samples from Saskatchewan potash mines.

---
## Torchaudio MFCC transform
- **sample_rate=48000**: The audio is sampled at 48 kHz.
- **n_fft=2048**: Each FFT window spans 2048 samples (~42.7ms).
- **hop_length=512**: The window moves 512 samples (~10.7ms) each time.
- **n_mels=128**: The Mel-scaled frequency bins are used.
- **n_mfcc=40**: The top 40 cepstral coefficients are retained.

Each 0.75-second waveform (36,000 samples) is processed into:
- An MFCC tensor of shape `[1, 40 , ~71]`.
- Flattened into a vector of shape `[2840]` for input into the MLP.

---

## Dataset

| Split | Samples | Description                           |
|-------|---------|---------------------------------------|
| Train | 3000    | Recorded roof impact sounds (labeled) |
| Test  | 309     | Held out for final evaluation         |

- **Label `0`** = drummy  
- **Label `1`** = tight  

---
## Accuracy
Accuracy on Test Set of 309 items was **94%**

## Model: Multi-Layer Perceptron (MLP)

I used a simple 2-layer MLP to classify the acoustic signals, 1 hidden layer with 512 neurons
- **Optimizer - Adam**
- **Loss - CE Loss with 2 final features**
- **LR = 0.0001**
- **Epochs = 10**


## 📊 Dataset

The dataset used in this project was published as part of the following open dataset:

> **Travis Wiens & Shahriar Islam (2022).**  
> **Data for evaluating mine roof stability via acoustic impact**.  
> [https://doi.org/10.1016/j.dib.2022.107854](https://doi.org/10.1016/j.dib.2022.107854)  
> [[ScienceDirect link](https://www.sciencedirect.com/science/article/pii/S235234092200066X?via%3Dihub)]

If you use this dataset, please cite the above paper.

---

## 📄 Reference Paper

This project is based on the methodology and analysis from:

> **Travis Wiens & Shahriar Islam (2021).**    
> **Using acoustic impacts and machine learning for safety classification of mine roofs**
> [https://doi.org/10.1016/j.ijrmms.2021.104912](https://doi.org/10.1016/j.ijrmms.2021.104912)  
> [[ScienceDirect link](https://www.sciencedirect.com/science/article/pii/S1365160921002963)]

---

## 🚀 How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/kanwarSINGH03/acoustic_classification.git
cd acoustic_classification
```

### 2. Folder Structure
### Download "mine_impact_data_2019.mat" from above given dataset link and save it like shown below.

``` bash
.
├── data/
│   └── mine_impact_data_2019.mat
├── dataset/
│   └── data.py           # Mine Dataset class
├── models/
│   └── models.py         # models classes
├── test.ipynb            # Jupyter Evaluation script
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
```
### 3. Python Setup
### Setup Python3 virtual enviroment
``` bash
python3 -m venv venv
```
### Install dependencies
``` bash
pip install -r requirements.txt
```

