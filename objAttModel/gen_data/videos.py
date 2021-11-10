from shape import Shape
from color import Color
from direction import Direction
from questions import create_questions
from PIL import Image, ImageDraw
import imageio
from random import choice, randint
import math

# We draw the image at a larger scale and then resize it down to get anti-aliasing
# This is necessary because PIL's draw methods don't anti-alias
IM_SIZE = 64
IM_DRAW_SCALE = 2
IM_DRAW_SIZE = IM_SIZE * IM_DRAW_SCALE

MIN_SHAPE_SIZE = IM_DRAW_SIZE / 8
MAX_SHAPE_SIZE = IM_DRAW_SIZE / 2

TRIANGLE_ANGLE_1 = 0
TRIANGLE_ANGLE_2 = -math.pi / 3

colors = list(Color)
shapes = list(Shape)
directions = list(Direction)

def create_video(filename, shape, color, direction, numFrames=10):
    r = randint(230, 255)
    g = randint(230, 255)
    b = randint(230, 255)
    rgb = (r, g, b)

    #speed = randint(1, MIN_SHAPE_SIZE)
    speed = 5

    if shape is Shape.RECTANGLE:
        w = randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        h = randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        x = randint(0, IM_DRAW_SIZE - w)
        y = randint(0, IM_DRAW_SIZE - h)
        xy = [(x, y), (x + w, y + h)]

    elif shape is Shape.CIRCLE:
        d = randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        x = randint(0, IM_DRAW_SIZE - d)
        y = randint(0, IM_DRAW_SIZE - d)
        xy = [(x, y), (x + d, y + d)]

    elif shape is Shape.TRIANGLE:
        s = randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        x = randint(0, IM_DRAW_SIZE - s)
        y = randint(math.ceil(s * math.sin(math.pi / 3)), IM_DRAW_SIZE)
        xy = [
          (x, y),
          (x + s * math.cos(TRIANGLE_ANGLE_1), y + s * math.sin(TRIANGLE_ANGLE_1)),
          (x + s * math.cos(TRIANGLE_ANGLE_2), y + s * math.sin(TRIANGLE_ANGLE_2)),
        ]

    else:
        raise Exception('Invalid shape!')

    #numFrames = 10
    frames = []
    for i in range(numFrames):
        frames.append(create_image(filename, shape, color, xy, rgb))
        xy = move_shape(xy, direction, speed)

    #frames[0].save(filename, format='GIF', append_images=frames[1:], save_all=True, duration=100, Loop=0)
    imageio.mimsave(filename, frames, duration=0.2)

def move_shape(xy, direction, dist):
    if direction is Direction.UP:
        for i in range(len(xy)):
            xy[i] = (xy[i][0], xy[i][1] - dist)

    elif direction is Direction.RIGHT:
        for i in range(len(xy)):
            xy[i] = (xy[i][0] + dist, xy[i][1])

    elif direction is Direction.DOWN:
        for i in range(len(xy)):
            xy[i] = (xy[i][0], xy[i][1] + dist)

    elif direction is Direction.LEFT:
        for i in range(len(xy)):
            xy[i] = (xy[i][0] - dist, xy[i][1])

    else:
        raise Exception('Invalid direction!')

    return xy

def create_image(filename, shape, color, xy, rgb):
    im = Image.new('RGB', (IM_DRAW_SIZE, IM_DRAW_SIZE), rgb)

    draw = ImageDraw.Draw(im)
    draw_shape(draw, shape, color, xy)
    del draw

    im = im.resize((IM_SIZE, IM_SIZE), resample=Image.BILINEAR)

    im.save(filename, 'png')

    return im

def draw_shape(draw, shape, color, xy):
    if shape is Shape.RECTANGLE:
        draw.rectangle(xy, fill=color.value)

    elif shape is Shape.CIRCLE:
        draw.ellipse(xy, fill=color.value)

    elif shape is Shape.TRIANGLE:
        draw.polygon(xy, fill=color.value)

    else:
        raise Exception('Invalid shape!')
