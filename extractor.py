import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def toTFIDF(df):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_counts = count_vect.fit_transform(df['source'])
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    header = count_vect.get_feature_names_out()
    X = X_tfidf.toarray()
    print(X.shape)
    df2 = pd.DataFrame(X, columns=header)
    return df2


def toVector(df):
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(df['source'])
    header = count_vect.get_feature_names_out()
    X = X_counts.toarray()
    print(X.shape)
    df2 = pd.DataFrame(X, columns=header)
    return df2


allFre = {}
def countVect(teks):
    kata_list = teks.split()
    indFre = {}
    for kata in kata_list:
        if kata in indFre:
            indFre[kata] += 1
        else:
            indFre[kata] = 1
        allFre[kata] = 0
    return indFre


def toVector2(df, all, source):
    header = []
    for kata, f in all.items():
        header.append(kata)
    df2 = pd.DataFrame(columns=header)

    for index, freq_dict in df.items():
        for kata, frekuensi in freq_dict.items():
            if kata in df2.columns:
                df2.at[index, kata] = frekuensi
    df2 = df2.fillna(0)
    print(df.shape)
    return df2


dfsc = pd.read_csv('sc.csv')
dfts = pd.read_csv('ts.csv')
sc2tfidf = toTFIDF(dfsc)
sc2tfidf.columns = ['st_' + col for col in sc2tfidf.columns]
sc2vector = toVector(dfsc)
sc2vector.columns = ['sv_' + col for col in sc2vector.columns]
ts2tfidf = toTFIDF(dfts)
ts2tfidf.columns = ['tt_' + col for col in ts2tfidf.columns]
ts2vector = toVector(dfts)
ts2vector.columns = ['tv_' + col for col in ts2vector.columns]

sc_ts2tfidf = pd.concat([sc2tfidf, ts2tfidf], axis=1)
sc_ts2vector = pd.concat([sc2vector, ts2vector], axis=1)
sc2tfidf_vector = pd.concat([sc2tfidf, sc2vector], axis=1)
ts2tfidf_vector = pd.concat([ts2tfidf, ts2vector], axis=1)

sc2tfidf['label'] = dfts['label']
sc2tfidf.to_csv("sc2tfidf.csv")
sc2vector['label'] = dfts['label']
sc2vector.to_csv("sc2vector.csv")

ts2tfidf['label'] = dfts['label']
ts2tfidf.to_csv("ts2tfidf.csv")
ts2vector['label'] = dfts['label']
ts2vector.to_csv("ts2vector.csv")

sc_ts2tfidf['label'] = dfts['label']
sc_ts2tfidf.to_csv("sc_ts2tfidf.csv")
sc_ts2vector['label'] = dfts['label']
sc_ts2vector.to_csv("sc_ts2vector.csv")

sc2tfidf_vector['label'] = dfts['label']
sc2tfidf_vector.to_csv("sc2tfidf_vector.csv")
ts2tfidf_vector['label'] = dfts['label']
ts2tfidf_vector.to_csv("ts2tfidf_vector.csv")

sc_ts2tfidf_vector = pd.concat([sc_ts2tfidf, sc_ts2vector], axis=1)
sc_ts2tfidf_vector['label'] = dfts['label']
sc_ts2tfidf_vector.to_csv("sc_ts2tfidf_vector.csv")
