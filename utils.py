import numpy as np
import pandas as pd


def load_dataset():
    train = pd.read_csv('dataset/sign_mnist_train.csv')
    test = pd.read_csv('dataset/sign_mnist_test.csv')
    
    train_y = np.array(train['label'].values)
    test_y = np.array(test['label'].values)
    
    classes = np.unique(train_y)
    
    train.drop('label', axis=1, inplace=True)
    test.drop('label', axis=1, inplace=True)
    
    train_images = train.values
    train_images = np.array([np.reshape(i, (28, 28)) for i in train_images])
    
    test_images = test.values
    test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
    
    return train_images, train_y, test_images, test_y, classes