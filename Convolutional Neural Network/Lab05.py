# Allan Barrantes B80986

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from ConvNN import ConvNN
from sklearn.metrics import confusion_matrix
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
# Custom subdirectory to find images
DIRECTORY = "images"

def load_data():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    classes = [n.decode('utf-8') for n in unpickle(DIRECTORY+"/batches.meta")[b'label_names']]
    x_train = None
    y_train = []
    for i in range(1,6):
        data = unpickle(DIRECTORY+"/data_batch_"+str(i))
        if i>1:
            x_train = np.append(x_train, data[b'data'], axis=0)
        else:
            x_train = data[b'data']
        y_train += data[b'labels']
    data = unpickle(DIRECTORY+"/test_batch")
    x_test = data[b'data']
    y_test = data[b'labels']
    return classes,x_train,y_train,x_test,y_test

def plot_tensor(tensor, perm=None):
    if perm==None: perm = (1,2,0)
    plt.figure()
    plt.imshow(tensor.permute(perm).numpy().astype(np.uint8))
    plt.show()
    

def fit(data, epochs = 110, learning_rate = 0.001, momentum = 0.9, decay = 0.0000000001):
    cnn = ConvNN().to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)    
    for epoch in range(epochs+1):
        for _, (imgs, labels) in enumerate(data):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = cnn(imgs)
            loss = loss_criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print (f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return cnn

classes, x_train, y_train, x_test, y_test = load_data()
x_train, x_test, y_train, y_test = torch.Tensor(x_train), torch.Tensor(x_test), torch.Tensor(y_train), torch.Tensor(y_test)
x_train, x_test = x_train.reshape(50000, 3, 32, 32 ), x_test.reshape(10000, 3, 32, 32 ) 
y_train, y_test = y_train.type(torch.LongTensor), y_test.type(torch.LongTensor) 
train_data = torch.utils.data.DataLoader(list(zip(x_train,y_train)), batch_size=batch_size)
test_data = torch.utils.data.DataLoader(list(zip(x_test,y_test)), batch_size=batch_size)


cnn = fit(train_data)

# Save model
PATH = './cnn2.pth'
torch.save(cnn.state_dict(), PATH)

# Get accuracy for test
with torch.no_grad():
    mypred = torch.empty(0, dtype=torch.int64)
    mypred = mypred.to(device)
    correct = 0
    samples = 0
    class_correct = [0 for i in range(10)]
    class_samples = [0 for i in range(10)]
    for images, labels in test_data:
        images = images.to(device)
        labels = labels.to(device)
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        mypred = torch.cat((mypred, predicted), 0)
        samples += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                class_correct[label] += 1
            class_samples[label] += 1
    accuracy = 100.0 * correct / samples
    print(f'The cnn accuracy is: {accuracy} %')
    for i in range(10):
        acc_class = 100.0 * class_correct[i] / class_samples[i]
        print(f'Accuracy of class {classes[i]} is: {acc_class} %')
    print(f'Confusion matrix: {confusion_matrix(y_test,mypred)}')
