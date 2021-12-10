from shape import Shape
from color import Color
from direction import Direction
from videos import create_video
from questions import create_questions
#from questions_orig import create_questions
from random import choice
import json
import os

if not os.path.exists('easy_vqa_video/data/train/videos'):
  os.makedirs('easy_vqa_video/data/train/videos/')
if not os.path.exists('easy_vqa_video/data/test/videos'):
  os.makedirs('easy_vqa_video/data/test/videos/')

colors = list(Color)
shapes = list(Shape)
directions = list(Direction)

NUM_TRAIN = 4000
NUM_TEST = 1000
NUM_FRAMES = 10

def create_data(video_path, num):
    qs = []
    for i in range(num):
        shape = choice(shapes)
        color = choice(colors)
        direction = choice(directions)

        create_video(f'{video_path}/{i}.gif', shape, color, direction, NUM_FRAMES)
        new_qs = create_questions(shape, color, direction, i)
        qs += new_qs
    return qs

train_questions = create_data('easy_vqa_video/data/train/videos', NUM_TRAIN)
test_questions = create_data('easy_vqa_video/data/test/videos', NUM_TEST)

all_questions = train_questions + test_questions
all_answers = list(set(map(lambda q: q[1], all_questions)))

with open('easy_vqa_video/data/train/questions.json', 'w') as file:
    json.dump(train_questions, file)
with open('easy_vqa_video/data/test/questions.json', 'w') as file:
    json.dump(test_questions, file)

with open('easy_vqa_video/data/answers.txt', 'w') as file:
  for answer in all_answers:
    file.write(f'{answer}\n')

print(f'Generated {NUM_TRAIN} train videos and {len(train_questions)} train questions.')
print(f'Generated {NUM_TEST} test videos and {len(test_questions)} test questions.')
print(f'{NUM_FRAMES} frames per video.')
print(f'{len(all_answers)} total possible answers.')
