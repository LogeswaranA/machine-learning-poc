from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the model
model = joblib.load('movie_sentiment_model.pkl') #Model that saved during training
vectorizer = joblib.load('tfidf_vectorizer.pkl') #vectorizer that we used during training

# Create the Flask app
app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the data from the request
    print(data)
    review = pd.Series(data['review'])  # Convert to pandas Series
        # Transform the text data to the format expected by the model
    review_transformed = vectorizer.transform(review)
    
    # Make prediction
    prediction = model.predict(review_transformed)
    print(" prediction[0]", prediction[0])
    # Return the prediction as a JSON response

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
