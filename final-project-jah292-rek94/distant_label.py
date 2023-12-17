import os
from nltk.tokenize import word_tokenize
from pathlib import Path
import json
import csv

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

dirs = [os.path.join('final_slides_json_distant', 'CS-0441 Lecture Slides'),
  os.path.join('final_slides_json_distant', 'CS-0449 Lecture Slides'),
  os.path.join('final_slides_json_distant', 'CS-1541 Lecture Notes'),
  os.path.join('final_slides_json_distant', 'CS-1550 Lecture Slides'),
  os.path.join('final_slides_json_distant', 'CS-1567 Lecture Notes'),
  os.path.join('final_slides_json_distant', 'CS-1622 Lecture Slides')]

def get_concept_set(filename):
	concept_set = set()
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			concept_set.add(row[0])
	return concept_set

def label_json(json):
	concept_set = get_concept_set(os.path.join(ROOT_DIRECTORY,"all_fields_concepts.csv"))
	for j, slides in enumerate(json):
		k = 0
		for i in range(len(slides['words'])):
			if k > 0:
				json[j]['labels'][i] = 'I'
				k -= 1
			candidates = [word_tokenize(item) for item in concept_set if item.startswith(slides['words'][i])]
			# print("Candidates", candidates)
			matches = [candidate for candidate in candidates if list(map(str.lower, candidate)) == list(map(str.lower, slides['words'][i:i+len(candidate)]))]
			if matches:
				print("Matches", matches)
				k = max(map(len, matches)) - 1
				json[j]['labels'][i] = 'B'
	return json			
				
 

if __name__ == "__main__":
	print("Num unique concepts:", len(get_concept_set(os.path.join(ROOT_DIRECTORY,"all_fields_concepts.csv"))))
	for dir in dirs:
		files = Path(dir).glob('**/*.json')
		for file in files:
			with open(file, 'rb') as f:
				nwdir = os.path.join(ROOT_DIRECTORY, "final_slides_json_distant", Path(dir).stem)
				print(nwdir)
				try:
					os.mkdir(nwdir)
				except Exception as e:
					pass
				print(file)
				text = f.read()
				processed = json.loads(text)
				pair = label_json(processed)
				with open(os.path.join(nwdir, Path(file).stem)+".json", 'w') as f2:
					f2.write(json.dumps(pair, indent=4))