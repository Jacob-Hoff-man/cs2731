import os
from pathlib import Path
import json

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

dirs = [os.path.join('json_pair_slides', 'CS-0441 Lecture Slides'),
  os.path.join('json_pair_slides', 'CS-0449 Lecture Slides'),
  os.path.join('json_pair_slides', 'CS-1541 Lecture Notes'),
  os.path.join('json_pair_slides', 'CS-1550 Lecture Slides'),
  os.path.join('json_pair_slides', 'CS-1567 Lecture Notes'),
  os.path.join('json_pair_slides', 'CS-1622 Lecture Slides')]

dirs = [os.path.join('json_pair_slides', 'CS-1541 Lecture Notes')]

def convert_to_pair(json_parr):
    json_pair = {}
    for parr in json_parr:
        try:
            json_pair[parr] = [{'word': json_parr[parr]['words'][i], 'label': json_parr[parr]['labels'][i]} for i in range(len(json_parr[parr]['words']))]
        except IndexError:
            print(parr)
            exit()
    return json_pair
def convert_to_parr(json_pair):
    json_parr = {}
    for pair in json_pair:
        json_parr[pair] = {'words': [json_pair[pair][i]['word'] for i in range(len(json_pair[pair]))], 'labels': [json_pair[pair][i]['label'] for i in range(len(json_pair[pair]))]}
    index = 0
    prev_blank = False
    json_parr_together = [{'words': [], 'labels': []}]
    for parr in json_parr:
        if len(json_parr[parr]['words']) == 0:
            if prev_blank:
                index+=1
                print(index)
                json_parr_together.append({'words': [], 'labels': []})
            prev_blank = not prev_blank
        print(index)
        json_parr_together[index]['words'].extend(json_parr[parr]['words'])
        json_parr_together[index]['labels'].extend(json_parr[parr]['labels'])                
    return json_parr_together


for dir in dirs:
    files = Path(dir).glob('**/*.json')
    for file in files:
        with open(file, 'rb') as f:
            nwdir = os.path.join(ROOT_DIRECTORY, "final_slides_json", Path(dir).stem)
            print(nwdir)
            try:
                os.mkdir(nwdir)
            except Exception as e:
                print(e)
                pass
            print(file)
            text = f.read()
            processed = json.loads(text)
            pair = convert_to_parr(processed)
            with open(os.path.join(nwdir, Path(file).stem)+".json", 'w') as f2:
                f2.write(json.dumps(pair, indent=4))
