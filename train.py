from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
from model import build_model
from prepare_data import setup
from visualize import plot_loss, plot_accuracy

from PIL import Image, ImageSequence
import numpy as np

# Support command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--big-model', action='store_true', help='Use the bigger model with more conv layers')
parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
args = parser.parse_args()

if args.big_model:
    print('Using big model')
if args.use_data_dir:
    print('Using data directory')

# Prepare data
train_X_vids, train_X_seqs, train_Y, test_X_vids, test_X_seqs, test_Y, vid_shape, vocab_size, num_answers, _, _, _ = setup(args.use_data_dir)

print('\n--- Building model...')
model = build_model(vid_shape, vocab_size, num_answers, args.big_model)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

print('\n--- Training model...')
history = model.fit(
        [train_X_vids, train_X_seqs],
        train_Y,
        validation_data=([test_X_vids, test_X_seqs], test_Y),
        shuffle=True,
        epochs=32,
        callbacks=[checkpoint],
)

plot_loss(history)
plot_accuracy(history)
