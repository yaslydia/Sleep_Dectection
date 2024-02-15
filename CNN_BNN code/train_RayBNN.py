import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import raybnn_python
from PIL import Image
import os
from torchvision import datasets, transforms,utils
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score


def train_raybnn(x_train, y_train, x_test, y_test):
    accuracy_values = []
    recall_values = []
    kappa_values = []
    if isinstance(x_train, torch.Tensor):
        Rey_train = x_train.cpu().numpy()

    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value) / (max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value) / (max_value - min_value)

    print(x_train.shape)
    print(x_test.shape)


    dir_path = "/tmp/"

    max_input_size = 256
    input_size = 256

    max_output_size = 4
    output_size = 4

    max_neuron_size = 2000

    batch_size = 1000
    traj_size = 1

    proc_num = 2
    active_size = 1000

    training_samples = 8280
    crossval_samples = 8280
    testing_samples = 3660

    # Format MNIST dataset
    train_x = np.zeros((input_size, batch_size, traj_size, training_samples)).astype(np.float32)
    train_y = np.zeros((output_size, batch_size, traj_size, training_samples)).astype(np.float32)

    for i in range(x_train.shape[0]):
        j = (i % batch_size)
        k = int(i / batch_size)
        train_x[:, j, 0, k] = x_train[i, :]
        train_y[:, j, 0, k] = y_train[i, :]
        # idx = y_train[i]
        # train_y[int(idx[0]), j, 0, k] = 1.0

    crossval_x = np.copy(train_x)
    crossval_y = np.copy(train_y)

    # Create Neural Network
    arch_search = raybnn_python.create_start_archtecture(
        input_size,
        max_input_size,

        output_size,
        max_output_size,

        active_size,
        max_neuron_size,

        batch_size,
        traj_size,

        proc_num,
        dir_path
    )

    sphere_rad = arch_search["neural_network"]["netdata"]["sphere_rad"]

    arch_search = raybnn_python.add_neuron_to_existing3(
        10,
        10000,
        sphere_rad / 1.3,
        sphere_rad / 1.3,
        sphere_rad / 1.3,

        arch_search,
    )

    arch_search = raybnn_python.select_forward_sphere(arch_search)

    raybnn_python.print_model_info(arch_search)

    stop_strategy = "STOP_AT_TRAIN_LOSS"
    lr_strategy = "SHUFFLE_CONNECTIONS"
    lr_strategy2 = "MAX_ALPHA"

    loss_function = "sigmoid_cross_entropy_5"

    max_epoch = 0
    stop_epoch = 100000
    stop_train_loss = 0.005

    max_alpha = 0.01

    exit_counter_threshold = 100000
    shuffle_counter_threshold = 200

    total_epochs = 2

    for epoch in range(total_epochs):
        max_epoch += 1
        # Train Neural Network
        arch_search = raybnn_python.train_network(
            train_x,
            train_y,

            crossval_x,
            crossval_y,

            stop_strategy,
            lr_strategy,
            lr_strategy2,

            loss_function,

            max_epoch + 1,
            stop_epoch + 1,
            stop_train_loss,

            max_alpha,

            exit_counter_threshold,
            shuffle_counter_threshold,

            arch_search
        )

        test_x = np.zeros((input_size, batch_size, traj_size, testing_samples)).astype(np.float32)
        test_y = np.zeros((output_size, batch_size, traj_size, testing_samples)).astype(np.float32)

        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i / batch_size)

            test_x[:, j, 0, k] = x_test[i, :]
            test_y[:, j, 0, k] = y_test[i, :]
            print("y_test shape:",y_test.shape)

        # Test Neural Network
        output_y = raybnn_python.test_network(
            test_x,

            arch_search
        )

        print("output_y:", output_y.shape)

        pred = []
        y_label = []
        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i / batch_size)

            sample = output_y[:, j, 0, k]
            print(sample)

            pred.append(np.argmax(sample))

        pred = [np.argmax(output_y[:, i % batch_size, 0, int(i/batch_size)]) for i in range(x_test.shape[0])]
        pred = np.array(pred)
        y_label = [np.argmax(test_y[:, i % batch_size, 0, int(i/batch_size)]) for i in range(x_test.shape[0])]
        y_label = np.array(y_label)
        print("y_label:", y_label.shape, "pred.shape:", pred.shape)
        acc = accuracy_score(pred, y_label)

        ret = precision_recall_fscore_support(y_label, pred, average='macro')
        
        kappa = cohen_kappa_score(y_label, pred)
        
        print("Kappa Coefficient:", kappa)

        print("acc:", acc)
        print("ret:", ret)

        accuracy_values.append(acc)
        recall_values.append(ret[0])
        kappa_values.append(kappa)


    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_values, label='Accuracy', color='blue')
    plt.plot(recall_values, label='Recall', color='red')
    plt.plot(kappa_values, label='Kappa', color='green')
    plt.title('Model Performance over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
    plt.savefig("BNN_process_2.png")


    print(output_y.shape)
    return output_y.reshape(-1)




if __name__ == '__main__':
    features = np.load('/home/cxyycl/scratch/Microsleep-code/code/predictions/features.npy')
    labels = np.load('/home/cxyycl/scratch/Microsleep-code/code/predictions/labels.npy')
    features_val = np.load('/home/cxyycl/scratch/Microsleep-code/code/predictions/features_val.npy')
    labels_val = np.load('/home/cxyycl/scratch/Microsleep-code/code/predictions/labels_val.npy')
    #print(outputs_CNN.shape)
    output_y = train_raybnn(features, labels, features_val, labels_val)