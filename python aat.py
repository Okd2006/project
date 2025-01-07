import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle

data1=pd.read_csv('train.csv')
data1=np.array(data1)
m1, n1=data1.shape
data1=data1.T
Y1=data1[0]
X1=data1[1:n1]
X1=X1/255.

def init_params():
    W1=np.random.rand(70, 784)-0.5
    b1=np.random.rand(70, 1)-0.5
    W2=np.random.rand(10, 70)-0.5
    b2=np.random.rand(10, 1)-0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A=np.exp(Z)
    B=sum(np.exp(Z))
    return A/B

def forward_prop(W1, b1, W2, b2, X):
    Z1=W1.dot(X)+b1
    A1=ReLU(Z1)
    Z2=W2.dot(A1)+b2
    A2=softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_y=np.zeros((Y.size, 10))
    one_hot_y[np.arange(Y.size), Y]=1
    one_hot_y=one_hot_y.T
    return one_hot_y

def deri_ReLU(Z):
    return Z>0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_y=one_hot(Y)
    dZ2=A2-one_hot_y
    dW2=(1/m1)*dZ2.dot(A1.T)
    db2=(1/m1)*np.sum(dZ2)
    dZ1=W2.T.dot(dZ2)*deri_ReLU(Z1)
    dW1=(1/m1)*dZ1.dot(X.T)
    db1=(1/m1)*np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1=W1-alpha*dW1
    b1=b1-alpha*db1
    W2=W2-alpha*dW2
    b2=b2-alpha*db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions,  Y)
    return np.sum(predictions==Y)/Y.size

def gradient_decent(X, Y, alpha, iterations):
    W1, b1, W2, b2=init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2=forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2=back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2=update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i%10==0:
            print('Iteration:', i)
            print('Accuracy', get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

'''W1, b1, W2, b2=gradient_decent(X1, Y1, 0.1, 500)

with open('data.bin', 'wb') as f:
    pickle.dump(W1, f)
    pickle.dump(b1, f)
    pickle.dump(W2, f)
    pickle.dump(b2, f)'''

def make_prediction(X, W1, b1, W2, b2):
    _, _, _, A2=forward_prop(W1, b1, W2, b2, X)
    return A2, get_predictions(A2)

def test(W1, b1, W2, b2, X):
    A2, prediction=make_prediction(X, W1, b1, W2, b2)
    for i in range(10):
        print(i, float(A2[i])*100)
    print(prediction)
    '''current_image = X.reshape((28,  28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()'''