from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import json
import os
import numpy as np
import cv2
from PIL import Image, ImageSequence
import collections
#from easy_vqa import get_train_questions, get_test_questions, get_train_image_paths, get_test_image_paths, get_answers


def setup(use_data_dir):
    print('\n--- Reading questions...')
    if use_data_dir:
        # Read data from data/ folder
        def read_questions(path):
            with open(path, 'r') as file:
                qs = json.load(file)
            texts = [q[0] for q in qs]
            answers = [q[1] for q in qs]
            video_ids = [q[2] for q in qs]
            return (texts, answers, video_ids)
        train_qs, train_answers, train_video_ids = read_questions('data/train/questions.json')
        test_qs, test_answers, test_video_ids = read_questions('data/test/questions.json')
        #for i in range(5):
            #print(f'train_qs[{i}]: {train_qs[i]}')
            #print(f'train_answers[{i}]: {train_answers[i]}')
            #print(f'train_video_ids[{i}]: {train_video_ids[i]}')
        #quit()
    else:
        # Use the easy-vqa package
        train_qs, train_answers, train_video_ids = get_train_questions()
        test_qs, test_answers, test_video_ids = get_test_questions()
    print(f'Read {len(train_qs)} training questions and {len(test_qs)} testing questions.')

    print('\n--- Reading answers...')
    if use_data_dir:
        # Read answers from data/ folder
        with open('data/answers.txt', 'r') as file:
            all_answers = [a.strip() for a in file]
    else:
        # Read answers from the easy-vqa package
        all_answers = get_answers()
    num_answers = len(all_answers)
    print(f'Found {num_answers} total answers:')
    print(all_answers)

    print('\n--- Reading/processing videos...')
    def load_and_process_video(video_path):
        # Load video, then scale and shift pixel values to [-0.5, 0.5]
        video = Image.open(video_path)
        frames = np.array([np.array(frame.copy().convert('RGB').getdata(),dtype=np.uint8).reshape(frame.size[1],frame.size[0],3) for frame in ImageSequence.Iterator(video)])

        #print(frames.shape)
        return frames / 255 - 0.5

    def read_videos(paths):
        # paths is a dict mapping video ID to video path
        # Returns a dict mapping video ID to the processed video
        vids = {}
        for video_id, video_path in paths.items():
            vids[video_id] = load_and_process_video(video_path)
        #for i in range(0, len(paths), 2):
        #    vids[i][0] = load_and_process_video(video_path)
        #    vids[i][1] = load_and_process_video(video_path)
        print(type(vids))
        #quit()
        return vids

    def read_objects(paths, obj_id):
        objs = {}
        for video_id, video_path in paths.items():
            objs[video_id] = load_and_process_video(video_path+f'/{obj_id}.gif')
        return objs

    if use_data_dir:
        # Read videos from data/ folder
        def extract_paths(dir):
            paths = {}
            for filename in os.listdir(dir):
                if filename.startswith('video'):
                    video_id = int(filename[5:])
                    paths[video_id] = os.path.join(dir, filename)
            
            # return the paths in sorted order
            return collections.OrderedDict(sorted(paths.items()))

        #train_vids = read_videos(extract_paths('data/train/videos'))
        #test_vids  = read_videos(extract_paths('data/test/videos'))

        train_objects1 = read_objects(extract_paths('data/train/videos'), 0)
        train_objects2 = read_objects(extract_paths('data/train/videos'), 1)
        test_objects1  = read_objects(extract_paths('data/test/videos'), 0)
        test_objects2  = read_objects(extract_paths('data/test/videos'), 1)
    else:
        # Read images from the easy-vqa package #FIXME
        train_vids = read_images(get_train_image_paths())
        test_vids = read_images(get_test_image_paths())
    vid_shape = train_objects1[0].shape
    print(f'Read {len(train_objects1)} training videos and {len(test_objects1)} testing videos.')
    print(f'Each video has shape {vid_shape}.')

    print('\n--- Fitting question tokenizer...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_qs)

    # We add one because the Keras Tokenizer reserves index 0 and never uses it.
    vocab_size = len(tokenizer.word_index) + 1
    print(f'Vocab Size: {vocab_size}')
    print(tokenizer.word_index)

    print('\n--- Converting questions to bags of words...')
    train_X_seqs = tokenizer.texts_to_matrix(train_qs)
    test_X_seqs = tokenizer.texts_to_matrix(test_qs)
    print(f'Example question bag of words: {train_X_seqs[0]}')

    print('\n--- Creating model input images...')
    """
    train_X_first_objects = np.array([train_objects1[id] for id in train_video_ids])
    train_X_second_objects = np.array([train_objects2[id] for id in train_video_ids])
    test_X_first_objects = np.array([test_objects1[id] for id in test_video_ids])
    test_X_second_objects = np.array([test_objects2[id] for id in test_video_ids])
    """
    
    train_X_first_objects = []
    train_X_second_objects = []
    for i in range(len(train_objects1)):
        for j in range(4): # the number of questions (2 for each object)
            train_X_first_objects.append(train_objects1[i])
            train_X_second_objects.append(train_objects2[i])
    train_X_first_objects = np.array(train_X_first_objects)
    train_X_second_objects = np.array(train_X_second_objects)
    test_X_first_objects = []
    test_X_second_objects = []
    for i in range(len(test_objects1)):
        for j in range(4): # the number of questions (2 for each object)
            test_X_first_objects.append(test_objects1[i])
            test_X_second_objects.append(test_objects2[i])
    test_X_first_objects = np.array(test_X_first_objects)
    test_X_second_objects = np.array(test_X_second_objects)
    

    #train_X_vids = np.array([train_vids[id] for id in train_video_ids])
    #test_X_vids = np.array([test_vids[id] for id in test_video_ids])

    print('\n--- Creating model outputs...')
    train_answer_indices = [all_answers.index(a) for a in train_answers]
    test_answer_indices = [all_answers.index(a) for a in test_answers]
    train_Y = to_categorical(train_answer_indices)
    test_Y = to_categorical(test_answer_indices)
    print(f'Example model output: {train_Y[0]}')

    #print(f'all_answers: {all_answers}')
    #for i in range(8):
    #    print(f'train_video_ids[i]: {train_video_ids[i]}')
    #    print(f'train_Y[i]: {train_Y[i]}')

    return (train_X_first_objects, train_X_second_objects, train_X_seqs, train_Y, test_X_first_objects, test_X_second_objects,
            test_X_seqs, test_Y, vid_shape, vocab_size, num_answers,
            all_answers, test_qs, test_answer_indices)  # for the analyze script

