from flask import Flask, render_template, request, jsonify
import pandas as pd 
import numpy as np 
import re
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import random
import json
nltk.download('punkt')

app = Flask(__name__)

#clean des données
df = pd.read_csv('songs.csv')
df['Lyrics'] = df['Lyrics'].str.replace(r'.*Lyrics', 'Lyrics', regex=True)
mots_cles = ["Chorus", "Verse", "Pre-Chorus", "Bridge"]
masques = [df['Lyrics'].str.contains(mot) for mot in mots_cles]
masque_final = pd.concat(masques, axis=1).any(axis=1)
filtered_df = df[masque_final]
filtered_df.reset_index(drop=True, inplace=True)
def preprocess_text(text):
    text = text.replace('Lyrics', '')
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('\n', ' ')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
filtered_df['Lyrics'] = filtered_df['Lyrics'].apply(preprocess_text)
popularity_counts = filtered_df['Popularity'].value_counts()
df2 = filtered_df.copy()
high_popularity_songs = df2.loc[df2['Popularity'] > 90]
unique_high_popularity_songs = high_popularity_songs['Name'].unique()
df2 = df2.dropna()
cols_to_lower = ['Name', 'Artist', 'Album', 'Lyrics']
for col in cols_to_lower:
    df2[col] = df2[col].str.lower()
df2 = df2.drop_duplicates(keep='first').reset_index(drop=True)
missing_values = df2.isna().sum()
# Si vous souhaitez gérer ces valeurs manquantes (par exemple les supprimer), vous pouvez ajouter :
# df2.dropna(inplace=True)
unique_counts = df2.nunique()
percentage_unique = (df2.nunique() * 100) / len(df2)
doublons_songs = df2[df2.duplicated(subset=['Name'], keep='first')]
# Si vous souhaitez supprimer ces doublons, vous pouvez ajouter :
# df2.drop_duplicates(subset='Name', keep='first', inplace=True)
df2 = df2[(df2['Popularity'] >= 35) & (df2['Popularity'] <= 100)]
df2.reset_index(drop=True, inplace=True)

stemmer = PorterStemmer()
def token(txt):
    tokens = nltk.word_tokenize(txt)
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed_tokens)
df2_copy = df2.copy()
df2_copy['Lyrics'] = df2_copy['Lyrics'].apply(lambda x: token(x))
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidf.fit_transform(df2_copy['Lyrics'])
similarity = cosine_similarity(matrix)
def recommender(song_name):
    idx = df2_copy[df2_copy["Name"] == song_name].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    recommended_songs = [df2_copy.iloc[s_id[0]]["Name"] for s_id in distances[1:11]]
    return recommended_songs
recommended_songs_list = recommender('die for you')

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positif'
    elif analysis.sentiment.polarity < 0:
        return 'négatif'
    else:
        return 'neutre'
df2['Sentiment'] = df2['Lyrics'].apply(analyze_sentiment)

def recommend_songs_from_input(user_selections, df, num_recommendations=3):
    recommended_songs = []
    for selection in user_selections:
        user_favorite_songs = []
        if selection["Type"] == "Name":
            user_favorite_songs.append(selection["Value"])
        elif selection["Type"] == "Artist":
            artist_songs = df[df['Artist'] == selection["Value"]]['Name'].tolist()
            user_favorite_songs.extend(artist_songs)
        elif selection["Type"] == "Album":
            album_songs = df[df['Album'] == selection["Value"]]['Name'].tolist()
            user_favorite_songs.extend(album_songs)
        artist_selections = {}
        for song in user_favorite_songs:
            artist_name = df[df['Name'] == song]['Artist'].values[0]
            artist_selections[artist_name] = artist_selections.get(artist_name, 0) + 1
        most_selected_artist = max(artist_selections, key=artist_selections.get, default=None)
        if most_selected_artist:
            artist_songs = df[df['Artist'] == most_selected_artist]['Name'].tolist()
            potential_recs = [song for song in artist_songs if song not in user_favorite_songs]
            recommended_songs.extend(random.sample(potential_recs, min(num_recommendations, len(potential_recs))))
        if len(recommended_songs) < num_recommendations:
            sentiment_selections = [song_details['Sentiment'] for song in user_favorite_songs for _, song_details in df[df['Name'] == song].iterrows()]
            if sentiment_selections.count('positif') / len(sentiment_selections) >= 0.75:
                positive_songs = df[df['Sentiment'] == 'positif']['Name'].tolist()
                potential_recs = [song for song in positive_songs if song not in user_favorite_songs and song not in recommended_songs]
                recommended_songs.extend(random.sample(potential_recs, min(num_recommendations - len(recommended_songs), len(potential_recs))))
            elif sentiment_selections.count('négatif') / len(sentiment_selections) >= 0.75:
                negative_songs = df[df['Sentiment'] == 'négatif']['Name'].tolist()
                potential_recs = [song for song in negative_songs if song not in user_favorite_songs and song not in recommended_songs]
                recommended_songs.extend(random.sample(potential_recs, min(num_recommendations - len(recommended_songs), len(potential_recs))))
    recommended_songs = list({song: df[df['Name'] == song].iloc[0].to_dict() for song in recommended_songs}.values())
    return recommended_songs[:num_recommendations]

@app.route('/store_user_selection', methods=['POST'])
def store_user_selection():
    recommended_songs_json = request.json['user_selection']
    recommended_songs = json.loads(recommended_songs_json)
    recommended_songs_output = recommend_songs_from_input(recommended_songs)
    user_selection = pd.DataFrame(recommended_songs_output)
    return jsonify({'message': {'recommended_songs': recommended_songs_output}})

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
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

if __name__ == '__main__':
    app.run(debug=True)
