from shape import Shape
from color import Color
from direction import Direction
from videos import create_video
from questions import create_questions
#from questions_orig import create_questions
from random import choice
import json
import os

if not os.path.exists('data/train/videos'):
  os.makedirs('data/train/videos/')
if not os.path.exists('data/test/videos'):
  os.makedirs('data/test/videos/')

colors = list(Color)
shapes = list(Shape)
directions = list(Direction)

NUM_TRAIN = 4000
NUM_TEST = 1000
NUM_FRAMES = 10
NUM_SHAPES_PER_VID = 2

for i in range(NUM_TRAIN):
    os.makedirs(f'data/train/videos/video{i}')
for i in range(NUM_TEST):
    os.makedirs(f'data/test/videos/video{i}')

#lastShape = choice(shapes)
#lastColor= choice(colors)
#lastDirection = choice(directions)
def create_data(video_path, num):
    qs = []

    lastShape = choice(shapes)
    lastColor= choice(colors)
    lastDirection = choice(directions)
    for i in range(num):
        for j in range(NUM_SHAPES_PER_VID):
            if j == 0:
                shape = choice(shapes)
                lastShape = shape
                color = choice(colors)
                lastColor = color
                direction = choice(directions)
                lastDirection = direction

                create_video(f'{video_path}/video{i}/{j}.gif', shape, color, direction, NUM_FRAMES)
                new_qs = create_questions(shape, color, direction, i)
                qs += new_qs
            else:
                # ensure each object has a differnt shape, color, and direction than the other
                shape = choice([x for x in shapes if x != lastShape])
                color = choice([x for x in colors if x != lastColor])
                direction = choice([x for x in directions if x != lastDirection])

                create_video(f'{video_path}/video{i}/{j}.gif', shape, color, direction, NUM_FRAMES)
                new_qs = create_questions(shape, color, direction, i)
                qs += new_qs
    return qs

train_questions = create_data('data/train/videos', NUM_TRAIN)
test_questions = create_data('data/test/videos', NUM_TEST)

all_questions = train_questions + test_questions
all_answers = list(set(map(lambda q: q[1], all_questions)))

with open('data/train/questions.json', 'w') as file:
    json.dump(train_questions, file)
with open('data/test/questions.json', 'w') as file:
    json.dump(test_questions, file)

with open('data/answers.txt', 'w') as file:
  for answer in all_answers:
    file.write(f'{answer}\n')

print(f'Generated {NUM_TRAIN} train videos and {len(train_questions)} train questions.')
print(f'Generated {NUM_TEST} test videos and {len(test_questions)} test questions.')
print(f'{NUM_FRAMES} frames per video.')
print(f'{len(all_answers)} total possible answers.')
