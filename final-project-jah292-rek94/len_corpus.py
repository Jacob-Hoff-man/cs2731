import os
from pathlib import Path
import json

dirs = [os.path.join('json_slides', 'CS-0441 Lecture Slides'),
  os.path.join('json_slides', 'CS-0449 Lecture Slides'),
  os.path.join('json_slides', 'CS-1541 Lecture Notes'),
  os.path.join('json_slides', 'CS-1550 Lecture Slides'),
  os.path.join('json_slides', 'CS-1567 Lecture Notes'),
  os.path.join('json_slides', 'CS-1622 Lecture Slides')]

len_corpus = 0
total_annotated = 0
for dir in dirs:
	files = Path(dir).glob('**/*.json')
	for file in files:
		with open(file, 'rb') as f:
			text = f.read()
			processed = json.loads(text)
			len_corpus += len(processed)

dir = os.path.join('json_slides', 'CS-0441 Lecture Slides')
files = Path(dir).glob('**/*.json')
for file in files:
	with open(file, 'rb') as f:
		text = f.read()
		processed = json.loads(text)
		total_annotated += len(processed)

print("CS 0441 # lines: ", total_annotated)
dir = os.path.join('json_slides', 'CS-1567 Lecture Notes')
files = Path(dir).glob('**/*.json')
curr_annotated = -total_annotated
for file in files:
	with open(file, 'rb') as f:
		text = f.read()
		processed = json.loads(text)
		total_annotated += len(processed)

curr_annotated += total_annotated
print("CS 1567 # lines: ", curr_annotated)

curr_annotated = 0
total_class = 0
dir = os.path.join('json_slides', 'CS-1541 Lecture Notes')
files = Path(dir).glob('**/*.json')
for file in files:
	print(file)
	with open(file, 'rb') as f:
		print(file)
		text = f.read()
		processed = json.loads(text)
		if(file == Path(os.path.join('json_slides', 'CS-1541 Lecture Notes', 'CS1541_Lecture1.1_Introduction.json'))):
			total_annotated += len(processed)
			curr_annotated += len(processed)
			print("CS 1541 # lines: ", curr_annotated)
		total_class += len(processed)

print("CS 1541 # lines: ", curr_annotated)
print("% done:", curr_annotated/total_class)

curr_annotated = 0
total_class = 0
dir = os.path.join('json_slides', 'CS-0449 Lecture Slides')
files = Path(dir).glob('**/*.json')
for file in files:
	print(file)
	with open(file, 'rb') as f:
		text = f.read()
		processed = json.loads(text)
		if(file == Path(os.path.join('json_slides', 'CS-0449 Lecture Slides','01_CS449_Introduction.json')) or
	   file == Path(os.path.join('json_slides', 'CS-0449 Lecture Slides','02_CS449_Data-Representation.json'))):
			total_annotated += len(processed)
			curr_annotated += len(processed)
		total_class += len(processed)
	   
print("CS 0449 # lines: ", curr_annotated)
print("%% done:", curr_annotated/total_class)

percent_annotated = total_annotated / len_corpus

print(percent_annotated)
print(len_corpus)
print(total_annotated)
