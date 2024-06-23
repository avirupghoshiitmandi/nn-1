
# Digit Checker
A basic neural network of 2 hidden layers detecting digits between (0-9)

To run the nn first run the gradient.py which creates 6 matrices which is used when running the 2nd file MAIN.py


## Download train.csv and test.csv from kaggle

scroll down and click on download all which will then extract the .zip inside nn-1 repo so that the two .py files and the two .csv files are in same directory [click here](https://www.kaggle.com/competitions/digit-recognizer/data)


## Deployment

First change the 7th parameter in the following line of gradient.py file to your actual path of nn-1 repo 

```bash
 save_parameters(w1_F, b1_F, w2_F, b2_F, w3_F, b3_F,"PATH/OF/YOUR/REPO/nn-1")
```
now run the gradient.py which should give you W1.npy , W2.npy, W3.npy, B1.npy, B2.npy, B3.npy

now run the main file it should open to 2 tabs with tkinter app and image. Cross the image window to see different test examples.

if not installed before, install tkinter:-

```bash
 pip install tk
 ```
 or 
 ```bash
 pip3 install tk
 ```
