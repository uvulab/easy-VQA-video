from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv3D, MaxPooling3D, Flatten, Multiply, Add, Dropout
from tensorflow.keras.optimizers import Adam

def get_encoder(vid_shape, vocab_size, big_model):
    # The CNN
    vid_input = Input(shape=vid_shape)
    x1 = Conv3D(8, (2,3,3), padding='same')(vid_input)
    x1 = MaxPooling3D(pool_size=(1,2,2), padding='same')(x1)
    x1 = Conv3D(16, (2,3,3), padding='same')(x1)
    x1 = MaxPooling3D(pool_size=(2,2,2), padding='same')(x1)
    if big_model:
        x1 = Conv3D(32, 3, padding='same')(x1)
        x1 = MaxPooling3D(padding='same')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(32, activation='tanh')(x1)
    x1 = Dropout(0.4)(x1)

    # The question network
    q_input = Input(shape=(vocab_size,))
    x2 = Dense(32, activation='tanh')(q_input)
    x2 = Dense(32, activation='tanh')(x2)

    # Merge -> output
    out = Multiply()([x1, x2])
    out = Dense(32, activation='tanh')(out)

    return Model(inputs=[vid_input, q_input], outputs=out)

def get_score_model():
    embedding = Input(shape=(32,))
    x = Dense(32, activation='tanh')(embedding)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=embedding, outputs=x)


def build_model(vid_shape, vocab_size, num_answers, big_model):
    # Get fused feature embeddings
    vid_input1 = Input(shape=vid_shape)
    vid_input2 = Input(shape=vid_shape)
    q_input = Input(shape=(vocab_size,))
    encoder = get_encoder(vid_shape, vocab_size, big_model)
    v1_embedding = encoder([vid_input1, q_input])
    v2_embedding = encoder([vid_input2, q_input])

    # Get each object's score
    scoring_model = get_score_model()
    v1_score = scoring_model(v1_embedding)
    v2_score = scoring_model(v2_embedding)


    # Merge -> output
    v1_att = Multiply()([v1_embedding, v1_score])
    v2_att = Multiply()([v2_embedding, v2_score])
    out = Add()([v1_att, v2_att])
    out = Dense(32, activation='tanh')(out)
    out = Dense(num_answers, activation='softmax')(out)

    model = Model(inputs=[vid_input1, vid_input2, q_input], outputs=out)
    model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
