import requests
from bs4 import BeautifulSoup
import re

# URL and website information
root = "http://ohhla.com"
path = "/YFA_kendricklamar.html"
page = requests.get(root+path)

html = BeautifulSoup(page.text, "lxml")
all_songs = []

# song_tags are within each td with the following attributes
song_tags = html.find_all('td', attrs={'align': 'left', 'valign': 'top'})
for song in song_tags:
    children = song.findChildren()
    # add the child of a given song_tag if it isn't already in the all_songs list and it has a link to its lyrics
    all_songs.extend(((child.find(text=True), child['href']) for child in children if child.name == 'a' and child.find(text=True) not in dict(all_songs)))

# get only the a['href'] links to the lyrics of each songs
song_links = [songs[1] for songs in all_songs]

# store all lyrics in this list
lyrics = []

for song in song_links:
    #print(root+"/"+song)
    page = requests.get(root+"/"+song)
    #print(page.status_code)
    html = BeautifulSoup(page.text, "lxml")

    if (not html.select("pre")):
        lines = html.get_text()
        
    else:
        lines = html.select("pre")
        lines = lines[0].get_text()
        

    kendrick_lamar = []

    # Section out the lyrics by verse
    lyric_sections = re.split('\n\n', lines)
    
    # A flag that checks if we are reading a Kendrick lyric in the loop
    current_kendrick_lyric = False
    
    # Since section 0 contains information about the song and we only want lyrics, we skip section 0 and loop from section 1 to the last section
    for i in range(1, len(lyric_sections)):
        newline_index = lyric_sections[i].find('\n')
        # if the section contains the artist name "Kendrick Lamar" in it, add those lyrics and mark the current_kendrick_lyric flag True
        if "kendrick lamar" in lyric_sections[i][0:newline_index].lower():
            current_kendrick_lyric = True
            if newline_index != -1: # We want to make sure we're actually adding a section with Kendrick lyrics, and not a verse/chorus tag
                kendrick_lamar.append(lyric_sections[i][newline_index+1:])
        # if the section does not contain any artist name, check the current_kendrick_lyric flag to see whether or not to add the lyric
        elif not re.match('[\[\(]', lyric_sections[i][0:newline_index].lower()) and current_kendrick_lyric:
            kendrick_lamar.append(lyric_sections[i])
        # otherwise, ignore and mark the current_kendrick_lyric flag False
        else:
            current_kendrick_lyric = False
    lyrics.extend(kendrick_lamar) # save it all into lyrics list

# write lyrics list to a text file with each verse separated with newlines
with open('kendrick_lamar_lyrics.txt','w',encoding='utf8') as l:
#with open('kendrick_lamar_lyrics_no_preprocess.txt','w',encoding='utf-8') as l:
    for lyric in lyrics:
        # Preprocessing to get rid of special punctuation in the lyrics
        lyric = re.sub("\{?\*[A-Za-z0-9 \-\"]+\*\}?", "", lyric) # {*text*} or *text*
        lyric = re.sub("[\+\(\)\{\}\?\!\~\"]|[\.]+", "", lyric) # +, (, ), {, }, ?, !, ~, ", or 1 or more .
        lyric = lyric.strip().lower() # Strips preceding and trailing spaces and lowercases all words before the next regex
        lyric = re.sub(r"(\b[-',.:&\n]|[-',.:&\n]\b)|[\W_]", lambda x: (x.group(1) if x.group(1) else " "), lyric) # Removes punctuation. Preserves hyhenated words, contractions, commas, periods, colons (for timestamps), and newlines characters
        l.write(lyric+"\n\n")
