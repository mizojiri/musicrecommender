import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import csv

def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))  # ユークリッド距離を計算

def generate_playlist(userPreferences, data, generations=50):
    final_playlist = []  # 最終的なプレイリスト
    global_scaler = MinMaxScaler()
    
    # データをスケーリング
    id_column = data['id']
    data_features = data.drop(['id', 'type', 'uri', 'track_href', 'analysis_url',
                               'song_name', 'Unnamed: 0', 'title', 'genre'], axis=1)
    data_features = data_features[data_features.key != -1].drop_duplicates()
    scaled_data = pd.DataFrame(global_scaler.fit_transform(data_features), columns=data_features.columns)
    scaled_data['id'] = id_column

    userPreferences_features = userPreferences.drop(userPreferences.columns.difference(data_features.columns), axis=1)
    scaled_userPreferences = pd.DataFrame(global_scaler.transform(userPreferences_features), 
                                          columns=data_features.columns)

    for _, user_song in scaled_userPreferences.iterrows():
        best_song = None
        best_score = float('inf')
        sampled_songs = scaled_data.sample(10)  # ランダムに10曲を選択
        for generation in range(generations):
            # ユークリッド距離を計算してソート
            sampled_songs['score'] = sampled_songs.drop(['id'], axis=1).apply(
                lambda row: euclidean_distance(row, user_song), axis=1)
            sampled_songs = sampled_songs.sort_values(by='score')
            
            # 最も評価が高い5曲を残し、残りをランダムに入れ替え
            top_songs = sampled_songs.head(5)  # 最もスコアが良い5曲を選択
            replacement_songs = scaled_data.sample(5)  # ランダムに別の5曲を選択
            sampled_songs = pd.concat([top_songs, replacement_songs])  # 結合
        
        # 最終的に評価が最も高い曲をプレイリストに追加
        best_song = sampled_songs.iloc[0]
        final_playlist.append([best_song['id'], best_song['score']])
    
    return final_playlist

def save_playlist(playlist, filename="final_playlist.csv"):
    """プレイリストをCSVに保存"""
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'score'])
        writer.writerows(playlist)

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
        # 音楽データを読み込む
        data = pd.read_csv('genres_v2.csv', dtype={'song_name': 'str'})  # 音楽データのファイル名を指定
        # プレイリストを生成
        playlist = generate_playlist(userPreferences, data)
        # プレイリストをCSVに保存
        save_playlist(playlist)

if __name__ == "__main__":
    args = sys.argv
    main(args)
