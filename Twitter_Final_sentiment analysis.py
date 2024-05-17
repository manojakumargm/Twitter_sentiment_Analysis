import pandas as pd
import numpy as np
from sklearn.svm import SVC
import nltk

df_emoji = pd.read_csv("Emoji_Sentiment_Data.csv", usecols = ['Emoji', 'Negative', 'Neutral', 'Positive'])

# compare the polarity of the dataset and turn the polarity to binary
# 0 = negative, 1= positive
polarity_ls = []
for index, row in df_emoji.iterrows():
    
    # polarity == sentiment
    # initial polarity is negative
    polarity = 0.5  # Set default to neutral

    if row['Positive'] > row['Negative']:
        polarity = 1
    elif row['Positive'] < row['Negative']:
        polarity = 0

    polarity_ls.append(polarity)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# defining the function
def analyze_sentiment(tweet):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(tweet)

    if sentiment_scores['compound'] >= 0.05:
        return "positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"
    
# create new emoji dataset
new_df_emoji = pd.DataFrame(polarity_ls, columns=['sentiment'])
new_df_emoji['emoji'] = df_emoji['Emoji'].values
new_df_emoji

df_posts = pd.read_csv("new_data_with_sentiment.csv")
new_data = pd.DataFrame()
new_data = df_posts



new_data.isnull().sum()
new_data['post'] = new_data['post'].str.lower()

import string

def remove_punctuation(text):
    # Check if the value is NaN
    if pd.isna(text):
        return text
    else:
        return text.translate(str.maketrans('', '', string.punctuation))

new_data['post'] = new_data['post'].apply(remove_punctuation)

def remove_stopwords(text):
    # Check if the value is NaN
    if pd.isna(text):
        return text
    else:
        words = nltk.word_tokenize(text)
        stopwrds = nltk.corpus.stopwords.words('english')
        new_list = [word for word in words if word not in stopwrds]
        return ' '.join(new_list)


new_data['post'] = new_data['post'].apply(remove_stopwords)
new_data

def perform_stemming(text):
    if pd.isna(text):
        return text
    else:
        stemmer = nltk.PorterStemmer()
        new_list = []
        words = nltk.word_tokenize(text)
        for word in words:
            new_list.append(stemmer.stem(word))
        return " ".join(new_list)
    
new_data['post'] = new_data['post'].apply(perform_stemming)



import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# TFIDF vectorizer
stopset = list(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True,strip_accents='ascii', stop_words=stopset)

e_c, p = 0, 0
for index, row in new_df_emoji.iterrows():
    p += 1 if row['sentiment'] else 0
    e_c += 1

new_df_post = new_data["post"]
new_df_post.fillna("")
y = new_data['sentiment']
# convert 'sentence' from text to features
X = vectorizer.fit_transform(new_data['post'])



# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=None)



classifier = SVC(kernel='linear',C=0.1)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(f"svc {accuracy_score(y_test,predictions)}")

from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier()
classifier1.fit(X_train, y_train)

predictions = classifier1.predict(X_test)
print(f"random {accuracy_score(y_test,predictions)}")

import emoji
def extract_text_and_emoji(text):
    global allchars, emoji_list
    # remove all tagging and links, not need for sentiments
    remove_keys = ('@', 'http://', '&', '#')
    clean_text = ' '.join(txt for txt in text.split() if not txt.startswith(remove_keys))
#     print(clean_text)
    
    # setup the input, get the characters and the emoji lists
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if  emoji.is_emoji(c)]
    
    # extract text
    clean_text = ' '.join([str for str in clean_text.split() if not any(i in str for i in emoji_list)])
    
    # extract emoji
    clean_emoji = ''.join([str for str in text.split() if any(i in str for i in emoji_list)])
    return (clean_text, clean_emoji)



def get_sentiment(s_input):
    # turn input into array
    input_array= np.array([s_input])
    # vectorize the input
    input_vector = vectorizer.transform(input_array)
    # predict the score of vector
    pred_senti = classifier1.predict(input_vector)

    return pred_senti[0]

def get_emoji_sentiment(emoji_ls, emoji_df=new_df_emoji):
    emoji_val_ls = []
    for e in emoji_ls:
        get_emo_senti = [row['sentiment'] for index, row in emoji_df.iterrows() if row['emoji'] == e]
        if get_emo_senti:
            emoji_val_ls.append(get_emo_senti[0])
        else:
            # Handle the case when the list is empty (no matching emoji found)
            emoji_val_ls.append(0)  # You might want to set a default value
    return emoji_val_ls




def get_text_emoji_sentiment(input_test):
    # separate text and emoji
    (ext_text, ext_emoji) = extract_text_and_emoji(input_test)
    print(f'\tExtracted: "{ext_text}" , {ext_emoji}')

    # get text sentiment
    if(len(ext_text)==0):
        senti_text = "neutral"
    else:
        senti_text = get_sentiment(ext_text)
    if(senti_text=="positive"):
        senti_value = 1
    elif(senti_text=="negative"):
        senti_value = 0;
    else:
        senti_value = 0.5 
        
    print(f'\tText value: {senti_text}')

    # get emoji sentiment
    senti_emoji_value = sum(get_emoji_sentiment(ext_emoji, new_df_emoji))
    print_emo_val_avg = 0.5 if len(ext_emoji) == 0 else senti_emoji_value/len(ext_emoji)
    print(f'\tEmoji average value: {print_emo_val_avg}')

    # avg the sentiment of emojis and text
    senti_avg = (senti_emoji_value + senti_value) / (len(ext_emoji) + 1)
    print(f'\tAverage value: {senti_avg}')

    # set value of avg sentiment to either pos or neg 
    if senti_avg > 0.6:
        senti_truth = "positive"
    elif senti_avg < 0.35:
        senti_truth = "negative"
    else:
        senti_truth = "neutral"
   
    
    return senti_truth

def get_sentiment_value_counts():
    return new_data['sentiment'].value_counts().to_dict()

print(get_text_emoji_sentiment("I can not decide if I like or hate this product."))

# Plot Bar Graph
import matplotlib.pyplot as plt

print(new_data['sentiment'].value_counts())
plt.figure(figsize=(8, 6))
new_data['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Analysis - Bar Graph')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Plot Pie Chart
plt.figure(figsize=(8, 8))
new_data['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'gray'])
plt.title('Sentiment Analysis - Pie Chart')
plt.show()
