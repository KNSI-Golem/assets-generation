import pandas as pd
import requests
import shutil
import re
import os

df_characters = pd.read_csv('Data/CharacterLevelData/Characters.csv')
encodings = pd.read_csv('Data/SkinLabelsEncodings.csv', index_col=0)

urls = df_characters['SkinUrl'].unique()
urls = ['https://' + url for url in urls]

for url in urls:
    skin_name = re.search('(?<=\/)[\w\-_.]*$', url).group()
    path = re.search('(?<=postacie\/).*$', url).group()[:-len(skin_name)]
    path = 'Data/Skins/' + path
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        with open(f'{path}/{skin_name}', 'wb') as f:
            resp.raw.decode_content = True
            shutil.copyfileobj(resp.raw, f)   
    
    
