from flask import Flask, render_template, request, jsonify
import pandas as pd 
import numpy as np 
import re
import nltk
import string
import re

app = Flask(__name__)

#clean des données
df = pd.read_csv('songs.csv')
df['Lyrics'] = df['Lyrics'].str.replace(r'.*Lyrics', 'Lyrics', regex=True)
mots_cles = ["Chorus", "Verse", "Pre-Chorus", "Bridge"]
masques = [df['Lyrics'].str.contains(mot) for mot in mots_cles]
masque_final = pd.concat(masques, axis=1).any(axis=1)
filtered_df = df[masque_final]
filtered_df.reset_index(drop=True, inplace=True)

nltk.download('punkt')

def preprocess_text(text):
    text = text.replace('Lyrics', '')
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('\n', ' ')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

filtered_df['Lyrics'] = filtered_df['Lyrics'].apply(preprocess_text)

dffinal = filtered_df

final = dffinal.dropna()

dffinal['Name'] = dffinal['Name'].str.lower()
dffinal['Artist'] = dffinal['Artist'].str.lower()
dffinal['Album'] = dffinal['Album'].str.lower()
dffinal['Lyrics'] = dffinal['Lyrics'].str.lower()

duplicate_rows = dffinal.duplicated(keep='first')
duplicates_df = dffinal[duplicate_rows]

dffinal = dffinal[~duplicate_rows]
dffinal = dffinal.reset_index(drop=True)

df2 = dffinal[(dffinal['Popularity'] >= 35) & (dffinal['Popularity'] <= 100)]
df2.reset_index(drop=True, inplace=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_term = request.form['search_term'].lower()
    search_result = df2[
        df2['Name'].str.lower().str.contains(search_term) |
        df2['Artist'].str.lower().str.contains(search_term) |
        df2['Album'].str.lower().str.contains(search_term)
    ]
    results = search_result.to_dict(orient='records')
    return jsonify(results)

@app.route('/')
def home():
    return "Accueil"

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

if __name__ == '__main__':
    app.run(debug=True)