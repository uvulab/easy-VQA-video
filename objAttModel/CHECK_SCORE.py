from keras.models import load_model, Model

from model import build_model
from prepare_data import setup


def check_scores(filename='model.h5'):
    # Prepare data
    train_X_first_objects, train_X_second_objects, train_X_seqs, train_Y, test_X_first_objects, test_X_second_objects, test_X_seqs, test_Y, vid_shape, vocab_size, num_answers, all_answers, _, _ = setup(True)

    # Load model
    #model = build_model(vid_shape, vocab_size, num_answers, args.big_model)
    model = load_model(filename)
    model.summary()
    score1_extractor = Model(inputs=model.input, outputs=model.get_layer('scoring_model').get_output_at(0))
    score2_extractor = Model(inputs=model.input, outputs=model.get_layer('scoring_model').get_output_at(1))
    scores1 = score1_extractor.predict([test_X_first_objects, test_X_second_objects, test_X_seqs])
    scores2 = score1_extractor.predict([test_X_first_objects, test_X_second_objects, test_X_seqs])
    print(scores1)
    print(scores2)

check_scores()
