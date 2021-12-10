from shape import Shape
from color import Color
from direction import Direction
from random import randint

USE_ALL_QS = False

def create_questions(shape, color, direction, video_id):
    shape_name = shape.name.lower()
    color_name = color.name.lower()
    direction_name = direction.name.lower()

    if USE_ALL_QS:
        questions = [
            (f'what direction does the {shape_name} move?', direction_name),
            (f'what direction does the {color_name} shape move?', direction_name),
            (f'what color is the shape that moves {direction_name}?', color_name),
            (f'what is the shape that moves {direction_name}?', shape_name),

            (f'what is the {color_name} shape?', shape_name),

            (f'what color is the {shape_name}?', color_name),
            (f'what is the color of the {shape_name}?', color_name),
        ]

        questions = list(filter(lambda _: randint(0, 99) < 32, questions))

    else:
        questions = [
            (f'what direction does the {shape_name} move?', direction_name),
            (f'what direction does the {color_name} shape move?', direction_name),
        ]

    return (map(lambda x: x + (video_id,), questions)) # video_ids? how to map?
