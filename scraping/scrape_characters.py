import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import threading
import time

df = pd.read_csv('Data/RankingLevelData/Players.csv')
base_url = 'https://new.margonem.pl'
df['Url'] = base_url + df['Url']

cnt = 0
global_lock = threading.Lock()
log = ''
l_new_players_data = []
l_new_characters_data = []

def scrape():
    global cnt
    global log
    characters_new_data = {'Id': [], 'Gender': [], 'Nickname': [],
                           'SkinUrl': [], 'Level': [], 'Profession': [],
                           'World': [], 'Guild': [], 'GuildId': [],
                           'PlayerId': []}
    players_new_data = {'AccountCreationDate': [], 'InGameDays': [],
                        'Rank': [], 'ForumPosts': [],
                        'Reputation': [], 'ReputationFactor': [],
                        'PlayerId': []}
    
    while True:
        global_lock.acquire()
        if cnt == len(df):
            global_lock.release()
            break
        elif not cnt % 100 and cnt != 0:
            portion_done = cnt / len(df)
            print('Progress: %.3f%%' % (portion_done * 100), end='   ')
            time_taken = time.time() - start_time
            eta = time_taken / portion_done # players_to_visit / frequency (after some transformations)
            eta_hours = int(eta // 3600)
            eta_minutes = int(eta % 3600 // 60)
            eta_seconds = int(eta % 60)
            print(f'ETA: {eta_hours} hours, {eta_minutes} minutes, {eta_seconds} seconds')
            
        row = df.iloc[cnt] 
        cnt += 1
        
        global_lock.release()
        
        resp = requests.get(row['Url'])
        soup = BeautifulSoup(resp.content, 'lxml')
        
        profile_header = soup.find('div', class_='profile-header-data-container')
        try:
            stats = profile_header.find_all('div', class_='label')
        except AttributeError:
            print('Error:', row['Url'])
            log += 'AttributeError - No Profile' + row['Url'] + '\n'
            
        players_new_data['AccountCreationDate'].append(stats[4].findNext().text)
        players_new_data['InGameDays'].append(stats[7].findNext().text)
        players_new_data['Rank'].append(stats[0].findNext().text)
        players_new_data['ForumPosts'].append(stats[2].findNext().text)
        players_new_data['Reputation'].append(stats[5].findNext().text)
        players_new_data['ReputationFactor'].append(stats[8].findNext().text)
        players_new_data['PlayerId'].append(row['Id'])
        
        public_characters = soup.find('div', class_='character-list')
        private_characters = None
        characters = []
        if public_characters:
            private_characters = public_characters.findNext('div', class_='character-list')
            public_characters = public_characters.find('ul').find_all('li')
            characters.extend(public_characters)
        if private_characters:
            private_characters = private_characters.find('ul')
            private_characters = private_characters.find_all('li')
            characters.extend(private_characters)
        
        for character in characters:
            characters_new_data['Id'].append(int(character['data-id']))
            characters_new_data['Nickname'].append(character['data-nick'])
            style = character.find('span')['style']
            skin_url = re.search("(?<=url\('\/\/).*'\);", style).group()[:-3]
            characters_new_data['SkinUrl'].append(skin_url)
            
            characters_new_data['Level'].append(character.find('input', class_='chlvl')['value'])
            characters_new_data['Profession'].append(character.find('input', class_='chprofname')['value'])
            characters_new_data['World'].append(character.find('input', class_='chworld')['value'])
            characters_new_data['Guild'].append(character.find('input', class_='chguild')['value'])
            characters_new_data['GuildId'].append(character.find('input', class_='chguildid')['value'])
            characters_new_data['Gender'].append(character.find('input', class_='chgender')['value'])
            characters_new_data['PlayerId'].append(row['Id'])
        
    l_new_players_data.append(players_new_data)
    l_new_characters_data.append(characters_new_data)


start_time = time.time()
thread_list = []
for _ in range(50):
    thread_list.append(threading.Thread(target=scrape))
    
# Starts threads
for thread in thread_list:
    thread.start()
    
for thread in thread_list:
    thread.join()
    
print('Scraping Done!')

l_new_players_data = [pd.DataFrame(data) for data in l_new_players_data]
l_new_characters_data = [pd.DataFrame(data) for data in l_new_characters_data]
df_new_players_data = pd.concat(l_new_players_data)
df_new_characters_data = pd.concat(l_new_characters_data)

for df in (df_new_players_data, df_new_characters_data):
    for column in df.columns:
        if str(df[column].dtype) == 'object':
            df[column] = df[column].str.replace('^\s+', '')
            df[column] = df[column].str.replace('\s+$', '')

df_new_characters_data['Gender'] = df_new_characters_data['Gender'].\
    map({'m': 'Male', 'k': 'Female'})
    
for col in ('InGameDays', 'ForumPosts', 'Reputation', 'ReputationFactor'):
    df_new_players_data[col] = df_new_players_data[col] .str.replace('\s', '')
df_new_players_data = df_new_players_data.astype({'InGameDays': 'int32',
                                                  'ForumPosts': 'int32',
                                                  'Reputation': 'int32',
                                                  'ReputationFactor': 'float32'})
df_new_players_data['AccountCreationDate'] = pd.to_datetime(df_new_players_data['AccountCreationDate'])

df_new_players_data.to_csv('Data/CharacterLevelData/Players.csv', index=0)
df_new_characters_data.to_csv('Data/CharacterLevelData/Characters.csv', index=0)
