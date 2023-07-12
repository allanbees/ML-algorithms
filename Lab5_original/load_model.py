# Script to load and execute the trained model

import torch
from ConvNN import ConvNN
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

# Load the saved model
cnn = ConvNN()
cnn.load_state_dict(torch.load('./cnn.pth'))

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


classes, x_train, y_train, x_test, y_test = load_data()
x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)
x_test = x_test.reshape(10000, 3, 32, 32 ) 
y_test = y_test.type(torch.LongTensor)

test_loader = torch.utils.data.DataLoader(list(zip(x_test,y_test)), batch_size=4)

# Get accuracy for test
with torch.no_grad():
    mypred = torch.empty(0, dtype=torch.int64)
    correct = 0
    samples = 0
    class_correct = [0 for i in range(10)]
    class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        mypred = torch.cat((mypred, predicted), 0)
        samples += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(4):
            label = labels[i]
            pred = predicted[i]
            print(pred)
            if (label == pred):
                class_correct[label] += 1
            class_samples[label] += 1

accuracy = 100.0 * correct / samples
print(f'The cnn accuracy is: {accuracy} %')
for i in range(10):
    acc_class = 100.0 * class_correct[i] / class_samples[i]
    print(f'Accuracy of class {classes[i]} is: {acc_class} %')
print(f'Confusion matrix: \n{confusion_matrix(y_test,mypred)}')