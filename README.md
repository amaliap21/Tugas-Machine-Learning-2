# Tugas-Machine-Learning-2

Machine Learning Major Assignment II for the course IF3270 Machine Learning aims to provide students with insights into how to implement Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) from scratch.

In the src folder, you’ll find Jupyter notebooks grouped by the model being used. The src/layers folder contains all the implemented layers. The bins folder is used to store models and evaluation graphs/images.

## Setup

### Create the virtual environment

```
python -m venv .venv
```

### Enter the virtual environment

#### On Windows (PowerShell):

```
.\.venv\Scripts\Activate.ps1
```

#### On Linux/MacOS:

```
source .venv/bin/activate
```

### Install dependencies

```
pip install -r requirements.txt
```

### Save newly added dependencies

```
pip freeze > requirements.txt
```
## How to Run
To view the evaluation results, it is highly recommended to run the relevant notebooks:
- cnn.ipynb
- rnn.ipynb
- lstm.ipynb

## Task Distribution

| Nama                                   | Tugas                                                            |
| -------------------------------------- | ---------------------------------------------------------------- |
| Amalia Putri (13522042)                | CNN (Conv2D), Pooling, Flatten, Dropout                          |
| Venantius Sean Ardi Nugroho (13522078) | Text Vectorization Layer, Embedding, SimpleRNN, and tech report |
| Julian Chandra Sutadi (13522080)       | LSTM, Bidirectional, Dense, notebook template                    |
