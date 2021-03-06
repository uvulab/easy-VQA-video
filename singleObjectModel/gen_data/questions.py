from shape import Shape
from color import Color
from direction import Direction
from random import randint

USE_ORIGINAL_QS = True

def create_questions(shape, color, direction, image_id):
    shape_name = shape.name.lower()
    color_name = color.name.lower()
    direction_name = direction.name.lower()

    if USE_ORIGINAL_QS:
        questions = [
            (f'what shape is in the image?', shape_name),
            (f'what shape is present?', shape_name),
            (f'what shape does the image contain?', shape_name),
            (f'what is the {color_name} shape?', shape_name),

            (f'what color is the {shape_name}?', color_name),
            (f'what is the color of the {shape_name}?', color_name),
            (f'what color is the shape?', color_name),
            (f'what is the color of the shape?', color_name),
        ]

        yes_no_questions = []

        for s in Shape:
            cur_shape_name = s.name.lower()
            pos_answer = 'yes' if s is shape else 'no'
            yes_no_questions.append((f'is there a {cur_shape_name}?', pos_answer))
            yes_no_questions.append((f'is there a {cur_shape_name} in the image?', pos_answer))
            yes_no_questions.append((f'does the image contain a {cur_shape_name}?', pos_answer))
            yes_no_questions.append((f'is a {cur_shape_name} present?', pos_answer))

            neg_answer = 'no' if s is shape else 'yes'
            yes_no_questions.append((f'is there not a {cur_shape_name}?', neg_answer))
            yes_no_questions.append((f'is there not a {cur_shape_name} in the image?', neg_answer))
            yes_no_questions.append((f'does the image not contain a {cur_shape_name}?', neg_answer))
            yes_no_questions.append((f'is no {cur_shape_name} present?', neg_answer))

        for c in Color:
            cur_color_name = c.name.lower()
            pos_answer = 'yes' if c is color else 'no'
            yes_no_questions.append((f'is there a {cur_color_name} shape?', pos_answer))
            yes_no_questions.append((f'is there a {cur_color_name} shape in the image?', pos_answer))
            yes_no_questions.append((f'does the image contain a {cur_color_name} shape?', pos_answer))
            yes_no_questions.append((f'is a {cur_color_name} shape present?', pos_answer))

            neg_answer = 'no' if c is color else 'yes'
            yes_no_questions.append((f'is there not a {cur_color_name} shape?', neg_answer))
            yes_no_questions.append((f'is there not a {cur_color_name} shape in the image?', neg_answer))
            yes_no_questions.append((f'does the image not contain a {cur_color_name} shape?', neg_answer))
            yes_no_questions.append((f'is no {cur_color_name} shape present?', neg_answer))

        questions = list(filter(lambda _: randint(0, 99) < 32, questions))
        questions.append(('what direction does the shape move?', direction_name))
        yes_no_questions = list(filter(lambda _: randint(0, 99) < 32, yes_no_questions))

    else:
        questions = [
            (f'what direction does the shape move?', direction_name)
        ]

    return (map(lambda x: x + (image_id,), questions))
