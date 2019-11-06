# MNIST-NeuralNet

## Dataset
### MNIST
Link: [Fashion-MNIST](https://www.kaggle.com/c/digit-recognizer/data)

## Software requirements
* Python 3.6, numpy, matplotlib, scikit-learn.

## Training
* Open file mnist_ann.ipynb and run on colab
* Or run file mnist_ann.py with command:

python mnist_ann.py --path=[str] –lr=[int] -epochs=[int] –batch_size=[int]

## Results
Model traing on training set 42,000 samples which devide into 33,600 training set and 8,400 valid set.

* Traning accuracy: 100.00000%

* Validation accuracy: 98.03571%

## Experiments
I build the model with one hidden layer and 10 neurals output with softmax. After that, I increase number of neurals of hidden layer to see the change of accuracy.
The table below show how accuracy change.

| Num Hidden Neurals | train_acc | val_acc |
| --- | --- | --- |
| 32 | 99.95833% | 96.36905% |
| 64 | 100.00000% | 97.39286% |
| 256 | 100.00000% | 97.94048% |
| 1000 | 100.00000% | 98.03571% |

