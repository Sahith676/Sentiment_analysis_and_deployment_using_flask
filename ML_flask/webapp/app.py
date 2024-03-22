from flask import Flask,render_template,request
import joblib
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
    
    
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@app.route('/predict',methods = ["get","post"])



def predict():
    # Claeaning the text
    def preprocess(text):
        # Removing special characters and digits
        text = re.sub(r'[^a-zA-Z\s]','', text)
        
        # Converting into lower case

        text = text.lower()
        # Tokeninzing into words

        tokens = nltk.word_tokenize(text)
        #Removing the stop words and Reducing to its root word
        filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

        return ' '.join(filtered_tokens)

    text_ml = request.form["txt"]
    clean_text = preprocess(text_ml)

    model = joblib.load("model/naive_bayes.pkl")
    
    clean_text_array = np.array([clean_text])
    prediction = model.predict(clean_text_array)
    return render_template('predict.html',prediction = prediction)


if __name__ == "__main__":
    app.run(debug = True,host = "0.0.0.0")