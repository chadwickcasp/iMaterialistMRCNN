import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import os

from collections import defaultdict

def load_tfevent_data(path):
    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    # print(event_acc.Tags())

    training_accuracies = event_acc.Scalars('loss')
    validation_accuracies = event_acc.Scalars('val_loss')
    # print()
    # print(list(training_accuracies))
    # print(list(validation_accuracies))
    # print()
    steps = len(training_accuracies)
    return training_accuracies, validation_accuracies, steps


def combine_tfevent_data(tfevent_data1, tfevent_data2):
    training_accuracies1, validation_accuracies1, steps1 = tfevent_data1
    training_accuracies2, validation_accuracies2, steps2 = tfevent_data2
    training_accuracies = training_accuracies1+training_accuracies2
    validation_accuracies = validation_accuracies1+validation_accuracies2
    steps = steps1+steps2
    return training_accuracies, validation_accuracies, steps


def plot_tensorflow_log(tfevent_data, path, train_color, valid_color):
    training_accuracies, validation_accuracies, steps = tfevent_data

    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2] # value
        y[i, 1] = validation_accuracies[i][2]

    label = path.split('.')[1].split('/')[1]
    plt.plot(x, y[:,0], label='train '+label, color=train_color)
    plt.plot(x, y[:,1], label='valid '+label, color=valid_color)
    plt.scatter(x, y[:,0], color='black')
    plt.scatter(x, y[:,1], color='black')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend(loc='upper right', frameon=True)


if __name__ == '__main__':
    log_files = defaultdict(list)
    for parent, dirs, files in os.walk('./'):
        if 'fashion' in parent.split('/')[1][:7]:
            for file in files:
                if 'events' in file:
                    log_files[parent].append(parent + "/" + file)
    train_colors = pl.cm.Blues(np.linspace(0.25, 0.75, len(log_files)))
    valid_colors = pl.cm.Reds(np.linspace(0.25, 0.75, len(log_files)))

    for i, items in enumerate(log_files.items()):
        parent, filepaths = items
        print(filepaths)
        print(parent)
        filepaths = sorted(filepaths, key=lambda x: x.split('/')[2].split('.')[3])
        try:
            tfevent_data = load_tfevent_data(filepaths[0])
        except KeyError as e:
            print("Couldn't find the appropriate items in the tfevents file: "+log_file)

        for path in filepaths[1:]:
            try:
                tfevent_data2 = load_tfevent_data(path)
            except KeyError as e:
                print("Couldn't find the appropriate items in the tfevents file: "+log_file)
            tfevent_data = combine_tfevent_data(tfevent_data, tfevent_data2)

        plot_tensorflow_log(tfevent_data, parent, train_colors[i], valid_colors[i])

    plt.show()