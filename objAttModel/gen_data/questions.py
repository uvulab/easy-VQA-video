from shape import Shape
from color import Color
from direction import Direction
from random import randint

def create_questions(shape, color, direction, video_id):
    shape_name = shape.name.lower()
    color_name = color.name.lower()
    direction_name = direction.name.lower()

    questions = [
        (f'what direction does the {shape_name} move?', direction_name),
        (f'what direction does the {color_name} shape move?', direction_name),
    ]

    return (map(lambda x: x + (video_id,), questions)) # video_ids? how to map?
