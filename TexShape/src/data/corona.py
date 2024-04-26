# Basic packages
import json
import numpy as np
import pandas as pd

# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# For Data Preprocessing
import re
import string
import emoji
from langdetect import detect, LangDetectException

# Transformers
# from transformers import BertTokenizerFast
# from transformers import TFBertModel
# import tensorflow as tf
# from tensorflow import keras
# import tensorflow_addons as tfa

# # LIME
# from lime.lime_text import LimeTextExplainer

# Tqdm
from tqdm import tqdm

CITY_TO_COUNTRY_PATH = "TexShape/data/processed/corona/city_to_country.json"

# Seed
def main():
    # # Loading the Data

    df = pd.read_csv("./Corona_NLP_train.csv", encoding="ISO-8859-1")
    df_test = pd.read_csv("./Corona_NLP_test.csv")

    df.head()
    df.info()


    df["TweetAt"] = pd.to_datetime(df["TweetAt"], dayfirst=True)

    tweets_per_day = (
        df["TweetAt"]
        .dt.strftime("%m-%d")
        .value_counts()
        .sort_index()
        .reset_index(name="counts")
    )

    # Load city to country mapping json
    with open(CITY_TO_COUNTRY_PATH) as f:
        city_to_country = json.load(f)

    # Drop the NaN values in the Location column
    df = df.dropna(subset=["Location"])
    df_test = df_test.dropna(subset=["Location"])
    df.isna().sum(), df_test.isna().sum()

    # Make keys and values lower case in city_to_country
    city_to_country = {
        k.lower(): v.lower() for k, v in city_to_country.items() if v is not None
    }

    # Make every location lowercased and strip leading and trailing whitespace
    # And cast to string
    df["Location"] = df["Location"].str.lower().str.strip().astype(str)
    df_test["Location"] = df_test["Location"].str.lower().str.strip().astype(str)
    df.head()

    # Get rid of punctuations except , and . in the Location column
    # Preprocessing the Location column
    df.loc[:, "Location"] = df["Location"].apply(lambda x: re.sub(r"[^\w\s\,\.]", "", x))
    df_test.loc[:, "Location"] = df_test["Location"].apply(
        lambda x: re.sub(r"[^\w\s\,\.]", "", x)
    )
    # Get rid of the leading and trailing spaces in the Location column
    df.loc[:, "Location"] = df["Location"].apply(lambda x: x.strip())
    df_test.loc[:, "Location"] = df_test["Location"].apply(lambda x: x.strip())

    # ## Country Converter Function
    def city_to_country_converter(location):
        # Split the location into parts
        location = str(location)
        parts = [part.strip() for part in location.split(",")]

        # Check if the full location is in the dictionary
        if location in city_to_country:
            return city_to_country[location]

        # Check if the first part of the location is in the dictionary
        elif parts[0] in city_to_country:
            return city_to_country[parts[0]]

        # Check if the second part of the location is in the dictionary
        elif len(parts) > 1 and parts[1] in city_to_country:
            return city_to_country[parts[1]]

        # Check if any part of the location is in the dictionary
        for part in parts:
            if part in city_to_country:
                return city_to_country[part]

        # If no match is found, return None
        return None


    # # Function convert city to country
    # def city_to_country_converter(city):
    #     if city in city_to_country:
    #         return city_to_country[city]
    #     else:
    #         return None

    # 'Location' DataFrame
    df.loc[:, "Country"] = df["Location"].apply(lambda x: city_to_country_converter(x))
    df_test.loc[:, "Country"] = df_test["Location"].apply(
        lambda x: city_to_country_converter(x)
    )

    ## Important
    # # # Extended and more sophisticated country mapping

    # def sophisticated_annotate_country(location):
    #     """
    #     Annotate the country based on the location string with a more sophisticated approach.
    #     This function uses more detailed string matching and known location patterns to infer the country.
    #     More complex or ambiguous locations will be marked as None.
    #     """
    #     if pd.isna(location):
    #         return None

    #     location = location.lower().strip()

    #     # Extended mappings for known locations, including cities, states, and regions
    #     country_map = {
    #         'usa': 'United States',
    #         'us': 'United States',
    #         'united states': 'United States',
    #         'america': 'United States',
    #         'ny': 'United States',
    #         'new york': 'United States',
    #         'california': 'United States',
    #         'los angeles': 'United States',
    #         'san francisco': 'United States',
    #         'chicago': 'United States',
    #         'texas': 'United States',
    #         'florida': 'United States',
    #         'miami': 'United States',
    #         'washington, dc': 'United States',
    #         'seattle': 'United States',
    #         'boston': 'United States',
    #         'atlanta': 'United States',
    #         'las vegas': 'United States',
    #         'new jersey': 'United States',
    #         'philadelphia': 'United States',
    #         'houston': 'United States',
    #         'dallas': 'United States',
    #         'austin': 'United States',

    #         'uk': 'United Kingdom',
    #         'united kingdom': 'United Kingdom',
    #         'england': 'United Kingdom',
    #         'london': 'United Kingdom',
    #         'manchester': 'United Kingdom',
    #         'birmingham': 'United Kingdom',
    #         'liverpool': 'United Kingdom',
    #         'edinburgh': 'United Kingdom',
    #         'glasgow': 'United Kingdom',
    #         'wales': 'United Kingdom',
    #         'scotland': 'United Kingdom',
    #         'northern ireland': 'United Kingdom',

    #         'canada': 'Canada',
    #         'toronto': 'Canada',
    #         'montreal': 'Canada',
    #         'vancouver': 'Canada',
    #         'ottawa': 'Canada',
    #         'calgary': 'Canada',

    #         'australia': 'Australia',
    #         'sydney': 'Australia',
    #         'melbourne': 'Australia',
    #         'brisbane': 'Australia',
    #         'perth': 'Australia',

    #         'india': 'India',
    #         'new delhi': 'India',
    #         'mumbai': 'India',
    #         'bangalore': 'India',
    #         'chennai': 'India',
    #         'kolkata': 'India',

    #         # Additional mappings for other countries and cities can be added here
    #         # For example, 'paris': 'France', 'berlin': 'Germany', etc.
    #     }

    #     # Check each key in the country map for a match
    #     for key in country_map:
    #         if key in location:
    #             return country_map[key]
    #     #
    #     # If no mapping found, return None
    #     return None

    # # Apply the sophisticated annotation function to the Location column
    # location_data['Country'] = location_data['Location'].apply(sophisticated_annotate_country)

    # # Displaying the first few rows of the data with the updated country annotation
    # location_data.head()


    # ## Unique Location Data
    # Create a new df from the original df["Location"].value_counts()
    location_data = pd.DataFrame(df["Location"].value_counts()).reset_index()
    location_data["Country"] = location_data["Location"].apply(
        lambda x: city_to_country_converter(x)
    )
    location_data.to_csv("location_data.csv")
    location_data.head()

    df["Country"].value_counts()

    # ## Discard Not Found Countries
    df = df[df["Country"].notna()]
    df_test = df_test[df_test["Country"].notna()]

    tweets_per_country = df["Country"].value_counts().reset_index(name="counts")

    tweets_per_country.counts.sum()

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(
        x="Country",
        y="counts",
        data=tweets_per_country,
        edgecolor="black",
        ci=False,
        palette="Spectral",
    )
    plt.title("Tweets count by country")
    plt.xticks(rotation=70)
    plt.yticks([])
    ax.bar_label(ax.containers[0])
    plt.ylabel("count")
    plt.xlabel("")
    plt.show()


    # # Data Cleaning
    # Discard Irrelevant Columns
    df = df[["OriginalTweet", "Country", "Sentiment"]]
    df_test = df_test[["OriginalTweet", "Country", "Sentiment"]]

    # Make positive and extremely positive tweets as positive and negative and extremely negative tweets as negative
    df["Sentiment"] = df["Sentiment"].replace(
        {"Extremely Positive": "Positive", "Extremely Negative": "Negative"}
    )
    df_test["Sentiment"] = df_test["Sentiment"].replace(
        {"Extremely Positive": "Positive", "Extremely Negative": "Negative"}
    )
    # Drop the neutral tweets
    df = df[df["Sentiment"] != "Neutral"]
    df_test = df_test[df_test["Sentiment"] != "Neutral"]
    (
        df["Sentiment"].value_counts(normalize=True),
        df_test["Sentiment"].value_counts(normalize=True),
    )

    # Label positive tweets as 1 and negative tweets as 0
    df["Sentiment"] = df["Sentiment"].replace({"Positive": 1, "Negative": 0})
    df_test["Sentiment"] = df_test["Sentiment"].replace({"Positive": 1, "Negative": 0})

    # Label United States as 1 and other countries as 0
    df["Country"] = df["Country"].apply(lambda x: 1 if x == "united states" else 0)
    df_test["Country"] = df_test["Country"].apply(
        lambda x: 1 if x == "united states" else 0
    )
    (
        df.value_counts("Country", normalize=True),
        df_test.value_counts("Country", normalize=True),
    )


    # ## Text Cleaning
    # Text Cleaning
    # Remove emojis
    def strip_emoji(text):
        return emoji.demojize(text)


    # Remove punctuation and special characters
    def strip_all_entities(text):
        text = text.replace("\r", "").replace("\n", " ").replace("\n", " ").lower()
        text = re.sub(r"(?:\@|https?\://)\S+", "", text)
        text = re.sub(r"[^\x00-\x7f]", r"", text)
        banned_list = string.punctuation + "Ã" + "±" + "ã" + "¼" + "â" + "»" + "§"
        table = str.maketrans("", "", banned_list)
        text = text.translate(table)
        return text


    # Remove hashtags
    def clean_hashtags(tweet):
        new_tweet = " ".join(
            word.strip()
            for word in re.split("#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)", tweet)
        )
        new_tweet2 = " ".join(word.strip() for word in re.split("#|_", new_tweet))
        return new_tweet2


    # Remove special characters
    def filter_chars(a):
        sent = []
        for word in a.split(" "):
            if ("$" in word) or ("&" in word):
                sent.append("")
            else:
                sent.append(word)
        return " ".join(sent)


    # Remove extra spaces
    def remove_mult_spaces(text):
        return re.sub("\s\s+", " ", text)


    texts_new = []
    for t in df.OriginalTweet:
        texts_new.append(
            remove_mult_spaces(
                filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))
            )
        )

    texts_new_test = []
    for t in df_test.OriginalTweet:
        texts_new_test.append(
            remove_mult_spaces(
                filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))
            )
        )

    df["text_clean"] = texts_new
    df_test["text_clean"] = texts_new_test

    df.value_counts("Sentiment", normalize=True), df.value_counts("Country", normalize=True)

    # ## Cleaning Non-English Tweets
    # Assuming your DataFrame has a column named 'Tweet'
    # df['Language'] = df['text_clean'].apply(lambda x: detect(x) if x is not None else None)

    languages = []
    for text in tqdm(df["text_clean"]):
        try:
            languages.append(detect(text))
        except LangDetectException:
            languages.append(None)

    # Add language column to DataFrame
    df["Language"] = languages

    df.head()
    languages = []
    for text in tqdm(df_test["text_clean"]):
        try:
            languages.append(detect(text))
        except LangDetectException:
            languages.append(None)

    # Add language column to DataFrame
    df_test["Language"] = languages

    # Discard tweets that are not in English
    df = df[df["Language"] == "en"]
    df_test = df_test[df_test["Language"] == "en"]

    (
        df.value_counts("Sentiment", normalize=True),
        df.value_counts("Country", normalize=True),
        df_test.value_counts("Sentiment", normalize=True),
        df_test.value_counts("Country", normalize=True),
    )

    # ## Text Length
    text_len = []
    for text in df.text_clean:
        tweet_len = len(text.split())
        text_len.append(tweet_len)

    df["text_len"] = text_len

    text_len_test = []
    for text in df_test.text_clean:
        tweet_len = len(text.split())
        text_len_test.append(tweet_len)

    df_test["text_len"] = text_len_test

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="text_len", data=df[df["text_len"] < 10], palette="mako")
    plt.title("Training tweets with less than 10 words")
    plt.yticks([])
    ax.bar_label(ax.containers[0])
    plt.ylabel("count")
    plt.xlabel("")
    plt.show()

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="text_len", data=df_test[df_test["text_len"] < 10], palette="mako")
    plt.title("Test tweets with less than 10 words")
    plt.yticks([])
    ax.bar_label(ax.containers[0])
    plt.ylabel("count")
    plt.xlabel("")
    plt.show()

    df = df[df["text_len"] > 4]

    df_test = df_test[df_test["text_len"] > 4]

    # # Lemma​tization of the training part
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    token_lens = []

    for txt in df["text_clean"].values:
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens.append(len(tokens))

    max_len = np.max(token_lens)

    print(f"Max len: {max_len}")

    token_lens = []

    for i, txt in enumerate(df["text_clean"].values):
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens.append(len(tokens))
        if len(tokens) > 80:
            print(f"Index: {i}, Text: {txt}")

    df["token_lens"] = token_lens

    df = df.sort_values(by="token_lens", ascending=False)
    df.head(20)

    df = df.iloc[12:]
    df.head()

    df = df.sample(frac=1).reset_index(drop=True)

    # # Lemmatization of the test part
    token_lens_test = []

    for txt in df_test["text_clean"].values:
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens_test.append(len(tokens))

    max_len = np.max(token_lens_test)

    print(f"Maximum length in tokens: {max_len}")

    token_lens_test = []

    for i, txt in enumerate(df_test["text_clean"].values):
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens_test.append(len(tokens))
        if len(tokens) > 80:
            print(f"Index: {i}, Text: {txt}")

    df_test["token_lens"] = token_lens_test

    df_test = df_test.sort_values(by="token_lens", ascending=False)
    df_test.head(10)

    df_test = df_test.iloc[5:]
    df_test.head(3)

    df_test = df_test.sample(frac=1).reset_index(drop=True)

