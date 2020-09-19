import numpy as np
import pandas as pd
import pickle


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


def create_model(neighborhood_df):
    x = neighborhood_df['description']
    y = neighborhood_df['ClusterLabel']

    # split the data into test and training sets
    description_train, description_test, name_train, name_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # vectorize the description as bag of words so it can be used in the model
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=0.7)
    x_train = tfidf_vectorizer.fit_transform(description_train.values)
    x_test = tfidf_vectorizer.transform(description_test.values)

    # save the vectorizer
    pickle.dump(tfidf_vectorizer, open("vectorizer.pkl", "wb"))

    # next we encode the names of the neighborhoods for the model
    encoder = LabelEncoder()
    encoder.fit(name_train)
    y_train_encoded = encoder.transform(name_train)
    y_test_encoded = encoder.transform(name_test)
    num_classes = np.max(y_train_encoded) + 1
    y_train = utils.to_categorical(y_train_encoded, num_classes)
    y_test = utils.to_categorical(y_test_encoded, num_classes)

    # create lookup table csv to find predicted cluster
    d = pd.DataFrame({'class_id': y_train_encoded})
    f = pd.DataFrame({'cluster': name_train}).reset_index(drop=True)
    cluster_table = f.join(d).drop_duplicates().reset_index(drop=True)
    cluster_table.to_csv('cluster_table', index=False)

    batch_size = 32
    epochs = 10

    model = Sequential()
    model.add(Dense(512, input_shape=(x_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.1)

    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)
    print('Test accuracy:', score[1])

    y_pred = model.predict(x_test)

    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    print(pd.DataFrame(matrix))

    # save the model
    pickle.dump(model, open('model.pkl', 'wb'))

    print("model created")
