# Tugas-Machine-Learning-2
<<<<<<< HEAD

Tugas Besar II pada kuliah IF3270 Pembelajaran Mesin agar peserta kuliah mendapatkan wawasan tentang bagaimana cara mengimplementasikan Convolutional Neural Network (CNN) dan Recurrent Neural Network. Pada tugas ini, peserta kuliah akan ditugaskan untuk mengimplementasikan modul forward propagation CNN dan RNN from scratch.
=======
Pada folder src terdapat notebook - notebook yang isinya dikelompokkan berdasarkan model apa yang digunakan. Pada folder src juga terdapat folder layers yang isinya semua layer yang diimplementasikan. Folder bins digunakan untuk menyimpan model dan gambar grafik - grafik pengujian.
>>>>>>> bf4321580abe2b82568c6e5ec542602c59f40d26

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

| Nama                                   | Tugas                                                                                                |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Amalia Putri (13522042)                | FFNN class, forward & back propagation, training, testing, autodiff                                  |
| Venantius Sean Ardi Nugroho (13522078) | Text Vectorization Layer, Embedding, Simple RNN, dan dokumen                                         |
| Julian Chandra Sutadi (13522080)       | Visualisasi Weight Distribution, Visualisasi Weight Gradient Distribution, Laporan, Debug dan Fixing |
