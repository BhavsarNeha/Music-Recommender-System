import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv')
song_extra_info = pd.read_csv('song_extra_info.csv')

# Merge data
train = train.merge(songs, on='song_id', how='left')
train = train.merge(members, on='msno', how='left')
train = train.merge(song_extra_info, on='song_id', how='left')

test = test.merge(songs, on='song_id', how='left')
test = test.merge(members, on='msno', how='left')
test = test.merge(song_extra_info, on='song_id', how='left')

# Convert date columns
train['registration_init_time'] = pd.to_datetime(train['registration_init_time'], format='%Y%m%d')
train['expiration_date'] = pd.to_datetime(train['expiration_date'], format='%Y%m%d')
test['registration_init_time'] = pd.to_datetime(test['registration_init_time'], format='%Y%m%d')
test['expiration_date'] = pd.to_datetime(test['expiration_date'], format='%Y%m%d')

# Handle missing values
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# Example of creating a new feature
train['song_length_log'] = np.log1p(train['song_length'])
test['song_length_log'] = np.log1p(test['song_length'])

# Save the preprocessed data
train.to_csv('train_preprocessed.csv', index=False)
test.to_csv('test_preprocessed.csv', index=False)
