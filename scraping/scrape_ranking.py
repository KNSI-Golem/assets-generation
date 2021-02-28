import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

def delete_whitespace(s):
    return re.sub('^\s*', '', re.sub('\s*$', '', s))

if __name__ == '__main__':
    base_url = 'https://new.margonem.pl/'
    url = 'https://new.margonem.pl/ladder'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    rankings_urls = []
    rankings_urls = [div.find('div', class_='ranking-footer').find('a')['href']\
                     for div in soup.find_all('div', class_='ranking-container')]
    rankings_urls = [base_url + url for url in rankings_urls]
    est_count = 0
    players = {'Id': [], 'Url': [], 'Name': []}
    characters = {'Player_id': [], 'World': [], 'Level': [], 'Profession': [], 'PH': [], 'Url': []}
    ids = set()
    
    for url in rankings_urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        
        n_pages = soup.find('div', class_='total-pages').text
        n_pages = int(re.sub('\s', '', n_pages))
        print(n_pages, '->', n_pages * 100 - 50)
        est_count += n_pages * 100 - 50
        
        world_name = re.search('(?<=,).*$', url).group()
        for i in range(1, n_pages + 1):
            page_url = url + f'?page={i}'
            response = requests.get(page_url)
            soup = BeautifulSoup(response.content, 'lxml')
            
            ranking = soup.find('tbody').find_all('tr')
            for player in ranking:
               player = player.find_all('td', class_='table-borders')
               character_url = player[0].find('a')['href']
               player_name = delete_whitespace(player[0].text)
               player_url = re.sub('#char.*', '', character_url)
               player_id = int(re.search('(?<=view,)\d+', player_url).group())
               
               if player_id not in ids:
                   ids.add(player_id)
                   players['Id'].append(player_id)
                   players['Url'].append(player_url)
                   players['Name'].append(player_name)
            
               characters['Player_id'].append(player_id)
               characters['World'].append(world_name)
               characters['Level'].append(delete_whitespace(player[1].text))
               characters['Profession'].append(delete_whitespace(player[2].text))
               characters['PH'].append(delete_whitespace(player[3].text))
               characters['Url'].append(character_url)
            
    print('total:', est_count)
    df_players = pd.DataFrame(players)
    df_players.to_csv('Data/RankingLevelData/Players.csv', index=None)
    
    df_characters = pd.DataFrame(characters)
    df_characters.to_csv('Data/RankingLevelData/Characters.csv', index=None)
