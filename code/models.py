# model with dynamic embeddings
from keras.layers import InputLayer, Convolution1D, MaxPooling1D, Concatenate, Flatten, Dense, Dropout, Input
from keras.layers import Embedding
from keras.models import Model
from keras.callbacks import ModelCheckpoint

def dynamic_embeddings(num_layers, filter_lengths, nb_filters, n_classes, hidden_size=250):
    # some constants
    max_len_char = 140
    dropout_rate = 0.5
    
    # dynamic embeddings and more n-grams
    input_layer = (Input(name = 'input', shape=(max_len_char,)))

    # Dynamic embeddings: TensorShape([Dimension(None), Dimension(246), Dimension(140)])
    embed = Embedding(input_dim=246, output_dim=140)(input_layer)

    convs = []
    for i in range(num_layers):
        for ksize in filter_lengths:
            conv = (Convolution1D(filters=nb_filters, kernel_size=ksize, padding="valid", activation="relu",\
                                                     strides=1, name ='conv%d_%d' % (i, ksize))(embed))
            pool = MaxPooling1D(pool_size =max_len_char - ksize + 1, name='pool%d_%d' % (i, ksize))(conv)
            convs.append(pool)

    concat = Concatenate()(convs)
    flatten = Flatten()(concat)
    flatten.get_shape()

    hidden = Dense(hidden_size, activation="relu")(flatten)
    dropout = Dropout(rate=dropout_rate)(hidden)

    output = Dense(n_classes, activation='softmax')(dropout)

    model = Model(inputs=input_layer, outputs=output)
    return model

def static_embeddings(num_layers, filter_lengths, nb_filters, n_classes, hidden_size=250):
    # some constants
    max_len_char = 140
    dropout_rate = 0.5
    
    input_layer = (Input(name = 'input', shape=(max_len_char, 246)))#len(small_chars_set))))

    convs = []
    for i in range(num_layers):
        for j in filter_lengths:
            conv = (Convolution1D(filters=nb_filters, kernel_size=j, padding="valid", activation="relu",\
                                             strides=1, name ='conv%d_%d' % (i, j))(input_layer))
            pool = MaxPooling1D(pool_size =max_len_char - j + 1, name='pool%d_%d' % (i, j))(conv)
            convs.append(pool)

    concat = Concatenate()(convs)
    flatten = Flatten()(concat)
    flatten.get_shape()

    hidden = Dense(hidden_size, activation="relu")(flatten)
    dropout = Dropout(rate=dropout_rate)(hidden)

    output = Dense(10, activation='softmax')(dropout)

    model = Model(inputs=input_layer, outputs=output)

    
    
    
    
    
    
    
    

