from model import build_model
from prepare_data import setup


def check_score(filename='model.h5'):
    # Prepare data
    train_X_first_objects, train_X_second_objects, train_X_seqs, train_Y, test_X_first_objects, test_X_second_objects, test_X_seqs, test_Y, vid_shape, vocab_size, num_answers, all_answers, _, _ = setup(True)

    # Load model
    #model = build_model(vid_shape, vocab_size, num_answers, args.big_model)
    model = models.load_model(filename)
    score_extractor = models.Model(inputs=model.input, outputs=model.get_layer('scoring_model').output)
    scores = score_extractor.predict([test_X_first_objects, test_X_second_objects, test_X_seqs])
    print(scores)
