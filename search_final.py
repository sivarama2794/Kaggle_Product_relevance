from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math
import pandas as pd
import re
stemmer = PorterStemmer()
stop_words = stopwords.words('english')

trainsetdata = pd.read_csv('train_new.csv', encoding="ISO-8859-1")[:1000]
procdata = pd.read_csv('product_descriptions_new.csv')[:1000]
attributedata = pd.read_csv('attributes_new.csv')
branddata = attributedata[attributedata.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
alldocfreq = pd.merge(trainsetdata, procdata, how='left', on='product_uid')
alldocfreq = pd.merge(alldocfreq, branddata, how='left', on='product_uid')


def co_sine_sim(x,y):
    vector1 = eval(x)
    vector2 = eval(y)
    mod_vector2 = dict()
    norm_vector1 = getnormvector(vector1,vector2.keys())
    for key,value in vector2.items():
        mod_vector2[key] = (1 + math.log10(value))
    norm_vector2 = getnormvector(mod_vector2, vector2.keys())
    return calculatedotproduct(norm_vector2,norm_vector1)

def brand_value(brand,se):
    se_set = se.split(" ")
    for word in se_set:
        if word == brand:
            return True
    return False


def gettf(x):
    freq_vector = defaultdict(int)
    tf_vector = dict()
    word_set = x.split(" ")
    for word in word_set:
        freq_vector[word] += 1
    for word_tf,freq in freq_vector.items():
        tf_vector[word_tf] = (1 + math.log10(freq))
    return tf_vector

def getdf(x,column):
    word_set = x.split(" ")
    df_vector = dict()
    for word in word_set:
        if 'title' in column:
            if not(word in df_title_values.keys()):
                df_value1 = alldocfreq.product_title.map(lambda x: 1 if word in x else 0).sum()
                df_title_values[word] = math.log10(N_Value / df_value1)
            df_vector[word] = df_title_values[word]
        else:
            if not (word in df_desc_values.keys()):
                df_value2 = alldocfreq.product_description.map(lambda x: 1 if word in x else 0).sum()
                df_desc_values[word] = math.log10(N_Value / df_value2)
            df_vector[word] = df_desc_values[word]
    return df_vector

def getfidf(x,y):
    tf_vector = eval(x)
    idf_vector = eval(y)
    tf_idf_vector = dict()
    for key,value in tf_vector.items():
        tf_idf_vector[key] = value *idf_vector[key]
    return tf_idf_vector

def getnormvector(vector,se_set):
    euclidian_distance_sum = 0
    norm_vector = dict()
    for token, tf_idf_weight in vector.items():
        euclidian_distance_sum += math.pow(tf_idf_weight, 2)
    euclidian_distance = math.sqrt(euclidian_distance_sum)
    for se_term in se_set:
        normalized_tf_idf_weight = 0
        if se_term in vector.keys():
            normalized_tf_idf_weight = (vector[se_term] / euclidian_distance)
        norm_vector[se_term] = normalized_tf_idf_weight
    return norm_vector


def stemming(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
        s = s.lower()
        s = s.replace("  ", " ")
        s = s.replace(",", "")  # could be number / segment later
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("-", " ")
        s = s.replace("//", "/")
        s = s.replace("..", ".")
        s = s.replace(" / ", " ")
        s = s.replace(" \\ ", " ")
        s = s.replace(".", " . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x ", " xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xbi ")
        s = s.replace(" by ", " xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°", " degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v ", " volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  ", " ")
        s = s.replace(" . ", " ")

        s = s.lower()
        s = s.replace("toliet", "toilet")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("vinal", "vinyl")
        s = s.replace("vynal", "vinyl")
        s = s.replace("skill", "skil")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("rustoleum", "rust-oleum")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        s = s.replace("DeckOver", "deck over")

        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s
    else:
        return "null"

def calculatedotproduct(vector1, vector2):
    dotproduct = 0
    for key, value in vector1.items():
        if key in vector2.keys():
            dotproduct += value * vector2[key]
    return dotproduct

def basic_linear_regression(x, y):
    # Basic computations to save a little time.
    length = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    # Σx^2, and Σxy respectively.
    sum_x_squared = sum(map(lambda a: a * a, x))
    sum_of_products = sum([x[i] * y[i] for i in range(length)])
    # Magic formulae!
    a = (sum_of_products - (sum_x * sum_y) / length) / (sum_x_squared - ((sum_x ** 2) / length))
    b = (sum_y - a * sum_x) / length
    return a, b

alldocfreq['search_term'] = alldocfreq['search_term'].map(lambda x: stemming(x))
alldocfreq['search_term'].apply(lambda x: [item for item in x if item not in stop_words])
alldocfreq['product_title'] = alldocfreq['product_title'].map(lambda x: stemming(x))
alldocfreq['product_title'].apply(lambda x: [item for item in x if item not in stop_words])
alldocfreq['product_description'] = alldocfreq['product_description'].map(lambda x: stemming(x))
alldocfreq['product_description'].apply(lambda x: [item for item in x if item not in stop_words])
alldocfreq['brand'] = alldocfreq['brand'].map(lambda x: stemming(x))
N_Value = len(alldocfreq.index)
df_title_values = dict()
df_desc_values = dict()
alldocfreq['search_term_tf'] = alldocfreq['search_term'].map(lambda x: str(gettf(x)))
alldocfreq['product_title_tf'] = alldocfreq['product_title'].map(lambda x: str(gettf(x)))
alldocfreq['product_description_tf'] = alldocfreq['product_description'].map(lambda x: str(gettf(x)))
alldocfreq['product_title_df'] = alldocfreq['product_title'].map(lambda x: getdf(x,'title'))
alldocfreq['product_description_df'] = alldocfreq['product_description'].map(lambda x: getdf(x,'desc'))
alldocfreq['product_title_tf_idf'] = alldocfreq.apply(lambda row: getfidf(row['product_title_tf'], row['product_title_df']), axis=1)
alldocfreq['product_description_tf_idf'] = alldocfreq.apply(lambda row: getfidf(row['product_description_tf'], row['product_description_df']), axis=1)
alldocfreq['desc_co_sine'] = alldocfreq.apply(lambda row: co_sine_sim(row['product_description_tf_idf'], row['search_term_tf']), axis=1)
alldocfreq['title_co_sine'] = alldocfreq.apply(lambda row: co_sine_sim(row['product_title_tf_idf'], row['search_term_tf']), axis=1)
alldocfreq['brand_co_sine'] = alldocfreq.apply(lambda row: brand_value(row['brand'], row['search_term']), axis=1)
alldocfreq['cum_co_sine'] = alldocfreq.apply(lambda row: 0.25*row['desc_co_sine']+0.25*row['title_co_sine']+0.5*row['brand_co_sine'], axis=1)
a, b = basic_linear_regression(alldocfreq['cum_co_sine'], alldocfreq['relevance'])
alldocfreq['predicted_score'] = alldocfreq['cum_co_sine'].map(lambda x: (x*a)+b)
alldocfreq.to_csv('alldocfreq.csv')