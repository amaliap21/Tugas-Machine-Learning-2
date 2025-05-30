# Tugas-Machine-Learning-2
Pada folder src terdapat notebook - notebook yang isinya dikelompokkan berdasarkan model apa yang digunakan. Pada folder src juga terdapat folder layers yang isinya semua layer yang diimplementasikan. Folder bins digunakan untuk menyimpan model dan gambar grafik - grafik pengujian.

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
## Cara run
Untuk melihat hasil pengujian, sangat disarankan untuk run tiap notebook yang relevan yaitu : lstm.ipynb . tubes2-rnn.ipynb, serta cnn.ipynb

## Pembagian Tugas
| NIM      | Kontribusi               |
|----------|--------------------------|
| 13522042 | CNN                      |
| 13522078 | Embedding, RNN           |
| 13522080 | Bidirectional, CNN, LSTM |
