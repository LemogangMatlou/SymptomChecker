import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import streamlit as st 

st.set_page_config(page_title='SymptomChecker',initial_sidebar_state='auto')

st.subheader('Please type in your symptoms below')

df = pd.read_csv('data.csv')
df_des = pd.read_csv('symptom_Description.csv')
df_precautions = pd.read_csv('Precaution.csv')

df['disease_id'] = df['Disease'].factorize()[0]
category_id_df = df[['Disease','disease_id']].drop_duplicates().sort_values('disease_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['disease_id','Disease']].values)

tfidf = TfidfVectorizer(sublinear_tf=True,min_df=5,encoding='utf-8',ngram_range=(1,2),stop_words='english')
features = tfidf.fit_transform(df['Symptoms']).toarray()
labels = df['disease_id']

X_train, X_test, y_train, y_test = train_test_split(df['Symptoms'],df['Disease'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

input_text = st.text_area("Please type in your symptoms here",height=175)

submit_int = st.button("Submit My Symptoms")

if submit_int:
    prediction = clf.predict(count_vect.transform([input_text]))
    st.success(f'You are likey to have {prediction[0]}')
    des_df = pd.read_csv('symptom_Description.csv')
    des_df.index = des_df['Disease']
    description = des_df.loc[f'{prediction[0]}', 'Description']
    st.write(description)

    pre_df = pd.read_csv('Precaution.csv')
    pre_df.index = pre_df['Disease']
    precaution = pre_df.loc[f'{prediction[0]}', 'Precautions']
    st.markdown(f'{prediction[0]} Precautions')
    st.warning(precaution)