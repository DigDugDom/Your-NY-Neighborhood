import pandas as pd
import spacy
import unicodedata


# python -m spacy download en_core_web_sm

# first read csv and create dataframe of the data
def read_listings(listing_file):
    base_df = pd.read_csv(listing_file)

    # Now keep the 2 columns needed which is the neighborhood_cleansed and the neighborhood_overview
    neighborhood_df = base_df[['neighborhood_overview', 'neighbourhood_cleansed']].dropna().copy()
    neighborhood_df = neighborhood_df.reset_index(drop=True)
    neighborhood_df = neighborhood_df.rename(
        columns={"neighborhood_overview": "description", "neighbourhood_cleansed": "name"})
    return neighborhood_df


# This function will take a doc and clean it by removing unimportant characters and lemmatizing
def doc_cleaner(doc):
    # this is for removing special characters
    no_special_char_doc = unicodedata.normalize('NFD', doc) \
        .encode('ascii', 'ignore') \
        .decode("utf-8")
    # spacy library can be used to remove stopwords, punctuation, digits, and spaces
    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(no_special_char_doc)
    return ([token.lemma_.lower() for token in tokens
             if token.is_stop is not True
             and token.is_punct is not True
             and token.is_digit is not True
             and token.is_space is not True])


# This function will take in the dataframe and clean each doc using doc_cleaner
def data_cleaning(neighborhood_df):
    for i, doc in neighborhood_df['description'].iteritems():
        token = doc_cleaner(doc)
        rejoin = ' '.join(token)
        neighborhood_df['description'].iloc[i] = rejoin
    # return dataframe with no blanks
    removed_blanks = neighborhood_df.loc[neighborhood_df.description != '']
    removed_blanks.to_csv('../model/filtered_neighborhood.csv', index=False)
