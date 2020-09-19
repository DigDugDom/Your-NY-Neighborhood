from flask import Flask, render_template, request
import pickle
import pandas as pd
import keras

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Load model and vectorizer using pickle
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # load lookup table and neighborhood_df
    cl = pd.read_csv('cluster_table.csv')
    neighborhood_df = pd.read_csv('neighborhood_df.csv')

    if request.method == 'POST':
        # get description from user and convert it to list format for vecorization
        message = request.form['message']
        data = [message]
        # vectorize the message
        desc = vectorizer.transform(data).toarray()

        # predict the cluster and return list of neighborhoods in that cluster
        class_prediction = loaded_model.predict_classes(desc)
        cluster = int(cl.cluster[cl['class_id'] == class_prediction[0]])
        my_prediction = list(neighborhood_df.name[neighborhood_df['ClusterLabel'] == cluster].drop_duplicates())
        #my_prediction = ['']
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')

