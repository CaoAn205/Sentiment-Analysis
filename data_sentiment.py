from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


wordnet_lemmatizer = WordNetLemmatizer()

file = pd.read_csv(
    'C:/Users/HP/Desktop/GitHub/Data Analytics (learning and pratising)/Data-Ananlytics-practice/Data-Ananlytics-practice/sentiment_project/dataset01.csv',
    header=None)
df = pd.DataFrame(file)
df = df.drop(columns=0, axis=1)
df = df.drop([0])
df.columns = ["Users", "Comments"]
df = df.iloc[:4999] #Data shape is too large to process

#Clean tags(@switchfoot,...), links(http://..), punctuations, time and make lowercase
def clean_text_1(text):
    text = text.split(' ')
    cleaned_text = []
    for word in text:
        if "http" in word:
            text.remove(word)
        elif "@" in word:
            text.remove(word)
        else:
            cleaned_text = cleaned_text + [word]

    cleaned_text = ' '.join([word for word in cleaned_text])
    cleaned_text = re.sub('[(!#+*/?%$,-.;:)]', '', cleaned_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub('\d [ap]m', '', cleaned_text)
    return cleaned_text
df['Clean01'] = df['Comments'].apply(lambda x: clean_text_1(x)) #apply

#Reduce text contractions like: i'm -> i am, also numbers like: time, room, location doesn't affect sentiment -> clean it
def contraction_remover(sentence):
    sentence = sentence.replace("'s", " is")
    sentence = sentence.replace("can't", "cannot")
    sentence = sentence.replace("cant", "cannot")
    sentence = sentence.replace("won't", "will not")
    sentence = sentence.replace("'m", " am")
    sentence = sentence.replace("'ve", " have")
    sentence = sentence.replace("n't", " not")
    sentence = sentence.replace("idk", "i do not know")
    sentence = sentence.replace(" ive ", "i have")
    sentence = sentence.replace("'re", " are")
    sentence = sentence.replace(" cuz ", "because")
    sentence = sentence.replace("tell ya", "tell you")
    sentence = sentence.replace("goin'", "going")
    sentence = sentence.replace(" ur ", " your ")
    sentence = sentence.replace(" u ", " you ")
    sentence = sentence.replace("nah", "no")
    sentence = re.sub('\d', '', sentence)  # remove number
    return sentence
df['Clean02'] = df['Clean01'].apply(lambda x: contraction_remover(x)) #apply

#Text Lemmizations could minimize the number of columns in DTM format
def text_lemmi(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    origin_word = []
    for word in sentence_words:
        origin_word += [wordnet_lemmatizer.lemmatize(word, 'v')]

    sentence = ' '.join([w for w in origin_word])
    return sentence
df['Clean03'] = df['Clean02'].apply(lambda x: text_lemmi(x)) #apply

#Fast sentiment from TextBlob
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
def sentiment(polarity):
    if polarity > 0.01:
        return "Positive"
    elif polarity < -0.01:
        return "Negative"
    else: return "Neutral"
df['Sentiment'] = df['Clean02'].apply(lambda x: sentiment(getPolarity(x))) #apply

#Remove stop words and make a Document-Term Matrix -> for Naive Bayes algs later
token = RegexpTokenizer(r'[a-z]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
data_cv = cv.fit_transform(df['Clean03'])
data_dtm = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())
data_dtm.index = df['Users']
#data_dtm.to_csv('C:/Users/HP/Desktop/GitHub/Data Analytics (learning and pratising)/Data-Ananlytics-practice/Data-Ananlytics-practice/sentiment_project/data_dtm01.csv')

#Make a corpus file:
corpus_sentiment = pd.DataFrame(data = df['Sentiment'])
corpus_sentiment.index = df['Users']
#corpus_sentiment.to_csv('C:/Users/HP/Desktop/GitHub/Data Analytics (learning and pratising)/Data-Ananlytics-practice/Data-Ananlytics-practice/sentiment_project/corpus01.csv')

#Split data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(data_dtm, df['Sentiment'], test_size=0.25, random_state=5)

#Training Model:
Sentiment_01 = MultinomialNB()
Sentiment_01.fit(X_train, Y_train)
Sentiment_01.fit(X_test, Y_test)



