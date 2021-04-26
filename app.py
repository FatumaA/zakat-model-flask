from flask import Flask, request, jsonify
import sys
import joblib
import traceback
from knn import loadedModel, model_columns
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def main():
    return '<h1> Welcome to the Zakat API <h1/>'

@app.route('/api/v1/classify', methods =['POST'])
def classify():
    if loadedModel:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_, index = [0]))
            query = query.reindex(columns=model_columns, fill_value=0)

            classification = list(loadedModel.predict(query))

            return jsonify({'classification': str(classification)})

        except:

            return jsonify({'error': 'Something went wrong', 'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    loadedModel = joblib.load('zakatfinalized_model.sav') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)