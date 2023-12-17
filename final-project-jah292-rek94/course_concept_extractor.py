# old code

import os
from PyPDF2 import PdfReader 
import pandas as pd

def convert_df_to_csv(df, file_name):
    df.to_csv(file_name, index=False, header=True)

def convert_pdf_course_slides_to_string(path, debug=False):
    if debug:
        print('reading', path)
    reader = PdfReader(path)
    total_strings = []
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        page_text = page.extract_text()
        strings = page_text.split('\n')
        total_strings = total_strings + strings
    return total_strings
  
class CourseConceptExtractor():
    def __init__(self, data_dirs, debug=False):
        self.concept_labels_dir_path = 'final-project/course-concept-labels/'
        self.dir_names = {}
        self.course_strings = {}
        self.course_dfs = {}

        for data_dirs_key in data_dirs:
            if debug:
                print(data_dirs_key, 'data directories')
            self.dir_names[data_dirs_key] = os.listdir(data_dirs[data_dirs_key])
            for dir in self.dir_names[data_dirs_key]:
                if dir.startswith('.'):
                    continue
                if debug:
                    print(dir)
                file_names = os.listdir(data_dirs[data_dirs_key] + '/' + dir)
                dir_strings = []
                for file in file_names:
                    if file.startswith('.'):
                        continue
                    path = data_dirs[data_dirs_key] + '/' + dir + '/' + file
                    if path.endswith('.pdf'):
                        dir_strings = dir_strings + convert_pdf_course_slides_to_string(path, debug)
                self.course_strings[dir] = dir_strings        
            for course_name in self.course_strings:
                self.course_dfs[course_name] = pd.DataFrame(self.course_strings[course_name])
                self.course_dfs[course_name].columns = ['text']
                if debug:
                    print(self.course_dfs[course_name])
                convert_df_to_csv(
                    self.course_dfs[course_name],
                    self.concept_labels_dir_path + course_name + ' Concept Labels.csv'
                )