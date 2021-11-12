import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.math import confusion_matrix
from keras import backend as K
from skimage.transform import resize
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from prepare_data import setup

# print the frequencies of each answer
#print('Label frequencies:')
#(unique, counts) = np.unique(Y_test, return_counts=True)
#frequencies = np.asarray(counts)
#frequencies = np.reshape(frequencies, (frequencies.shape[0], 1))
#print(frequencies)

# Prepare data
train_X_first_objects, train_X_second_objects, train_X_seqs, train_Y, test_X_first_objects, test_X_second_objects, test_X_seqs, test_Y, vid_shape, vocab_size, num_answers, all_answers, _, _ = setup(args.use_data_dir)

model = load_model('model.h5')

Y_test = np.argmax(test_Y, axis=1)

y_pred = model.predict([test_X_first_objects, test_X_second_objects, test_X_seqs])
y_pred = np.argmax(y_pred,axis=1)
con_mat = confusion_matrix(labels=Y_test, predictions=y_pred).numpy()

# normalize rows
sum_of_rows = con_mat.sum(axis=1)
normalized_con_mat = con_mat / sum_of_rows[:, np.newaxis]

# clear the current firgure
plt.clf()
plt.cla()
plt.close()

sns.heatmap(normalized_con_mat, annot=True, cmap=plt.cm.Blues, xticklabels=all_answers, yticklabels=all_answers, fmt='.2f')
plt.tight_layout()
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
