import pdftotext
import os
from nltk.tokenize import word_tokenize
from pathlib import Path
import json

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Read PDF
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        #reader = PyPDF2.PdfReader(file)
        #text = "\n".join([page.extract_text() for page in reader.pages])
        pdf = pdftotext.PDF(file)
        text = "\n\n".join(pdf)
    print(text)
    return text

def dummy_json(text):
    count = 1
    json_dict = {}
    for line in str.split(text, sep='\n'):
        words = word_tokenize(line)
        json = {'words': words, 'labels': ['O' for i in range(len(words))]}
        json_dict[count] = json
        count += 1
    return json_dict

if __name__ == '__main__':
    
    directory = 'data'
    for dir in Path(directory).glob('*'):
        nwdir = os.path.join(ROOT_DIRECTORY, "json_slides", dir.stem)
        txdir = os.path.join(ROOT_DIRECTORY, "data_text", dir.stem)
        try:
            os.mkdir(nwdir)
        except:
            pass
        try:
            os.mkdir(txdir)
        except:
            pass
        files = Path(dir).glob('**/*.pdf')
        for file in files:
            print(file)
            text = read_pdf(file_path=file)
            with open(os.path.join(txdir, Path(file).stem)+".txt", 'w') as f:
                f.write(text)
            dummy_json_text = dummy_json(text)
            with open(os.path.join(nwdir, Path(file).stem)+".json", 'w') as f:
                f.write(json.dumps(dummy_json_text, indent=4))
            


