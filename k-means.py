import sys
import pandas as pd
import csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

def euclidean_distance(row1, row2):
    """ユークリッド距離を計算"""
    return np.sqrt(np.sum((row1 - row2) ** 2))

def music_recommender(userPreferences):
    # データの読み込み
    raw_data = pd.read_csv('genres_v2.csv', dtype={'song_name': 'str'})
    print(raw_data.shape)

    # 不要な列を削除
    training_data = raw_data.drop(['type', 'uri', 'track_href', 'analysis_url',
                                   'song_name', 'Unnamed: 0', 'title', 'genre'], axis=1, inplace=False)

    # データクリーニング
    training_data = training_data[training_data.key != -1]
    training_data.drop_duplicates(inplace=True)

    # 正規化の準備
    global_scalar = MinMaxScaler()
    id_column = training_data['id']
    training_data.drop(['id'], axis=1, inplace=True)
    global_scalar.fit(training_data)
    training_data = pd.DataFrame(global_scalar.transform(training_data),
                                 index=training_data.index,
                                 columns=training_data.columns)
    training_data['id'] = id_column

    # クラスタリング
    kmeans = KMeans(n_clusters=10)
    training_data_clustered = kmeans.fit(training_data.drop(['id'], axis=1, inplace=False))
    training_data["cluster"] = training_data_clustered.labels_

    # ユーザーの好みデータの準備
    userPreferences.drop(userPreferences.columns.difference(["danceability", "energy", "key", "loudness", "mode",
                                                             "speechiness", "acousticness", "instrumentalness",
                                                             "liveness", "valence", "tempo", "duration_ms",
                                                             "time_signature"]), axis=1, inplace=True)

    userPreferences = pd.DataFrame(global_scalar.transform(userPreferences),
                                   index=userPreferences.index,
                                   columns=userPreferences.columns)

    # single_playlist作成
    single_playlist = []
    fields = ["id", "cluster", "score", "danceability", "energy", "key", "loudness", "speechiness", 
              "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

    for i in range(len(userPreferences)):
        # ユーザーの好みを基準にクラスタを決定
        cluster_index = (training_data_clustered.predict(userPreferences.iloc[[i]]))[0]
        cluster_songs = training_data[training_data.cluster == cluster_index]

        # クラスタ内の曲からランダムに1曲選ぶ
        cluster_songs_sample = cluster_songs.sample()
        row = [cluster_songs_sample.iloc[0]['id'], cluster_index]

        # スコア（ユークリッド距離）を計算
        score = euclidean_distance(
            cluster_songs_sample.drop(['id', 'cluster'], axis=1).iloc[0],
            userPreferences.iloc[i]
        )
        row.append(score)

        # 特徴量を追加
        row.extend(cluster_songs_sample.iloc[0][["danceability", "energy", "key", "loudness", "speechiness", 
                                                 "acousticness", "instrumentalness", "liveness", "valence", "tempo"]])
        single_playlist.append(row)
        print(single_playlist[i])

    # single_playlist.csvを書き出し
    filename = "single_playlist.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)  # ヘッダーを書き込み
        csvwriter.writerows(single_playlist)  # データを書き込み

def main(args):
    if len(args) < 2:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    file_name = args[1]
    if not os.path.isfile(file_name):
        print("File does not exist")
        sys.exit()
    else:
        userPreferences = pd.read_csv(file_name)
        music_recommender(userPreferences)

if __name__ == "__main__":
    args = sys.argv
    main(args)
