from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv3D, MaxPooling3D, Flatten, Multiply, Add, Dropout
from tensorflow.keras.optimizers import Adam

def get_vid_encoder(vid_shape, big_model):
    # The CNN
    vid_input = Input(shape=vid_shape)
    x = Conv3D(8, 3, padding='same')(vid_input)
    x = MaxPooling3D(padding='same')(x)
    x = Conv3D(16, 3, padding='same')(x)
    x = MaxPooling3D(padding='same')(x)
    if big_model:
        x = Conv3D(32, 3, padding='same')(x)
        x = MaxPooling3D(padding='same')(x)
    x = Flatten()(x)
    x = Dense(32, activation='tanh')(x)
    x = Dropout(0.4)(x)

    out = Dense(32, activation='tanh')(x) #out

    return Model(inputs=vid_input, outputs=out, name='vid_encoder')

def get_question_encoder(vocab_size):
    # The question network
    q_input = Input(shape=(vocab_size,))
    x = Dense(32, activation='tanh')(q_input)
    out = Dense(32, activation='tanh')(x)

    return Model(inputs=q_input, outputs=out, name='question_encoder')

def get_score_model():
    vid_embedding = Input(shape=(32,))
    q_embedding = Input(shape=(32,))

    # Merge
    x = Multiply()([vid_embedding, q_embedding])

    # Score
    x = Dense(32, activation='tanh')(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[vid_embedding, q_embedding], outputs=out, name='scoring_model')


def build_model(vid_shape, vocab_size, num_answers, big_model):
    # Get video embeddings
    vid_input1 = Input(shape=vid_shape)
    vid_input2 = Input(shape=vid_shape)
    vid_encoder = get_vid_encoder(vid_shape, big_model)
    v1_embedding = vid_encoder(vid_input1)
    v2_embedding = vid_encoder(vid_input2)

    # Get question embeddings
    q_input = Input(shape=(vocab_size,))
    q_encoder = get_question_encoder(vocab_size)
    q_embedding = q_encoder(q_input)

    # Get each object's score
    scoring_model = get_score_model()
    v1_score = scoring_model([v1_embedding, q_embedding])
    v2_score = scoring_model([v2_embedding, q_embedding])


    # Merge -> output
    v1_att = Multiply()([v1_embedding, v1_score])
    v2_att = Multiply()([v2_embedding, v2_score])
    out = Add()([v1_att, v2_att])
    out = Dense(32, activation='tanh')(out)
    out = Dense(num_answers, activation='softmax')(out)

    model = Model(inputs=[vid_input1, vid_input2, q_input], outputs=out)
    model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
