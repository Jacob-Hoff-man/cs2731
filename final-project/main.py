from course_concept_extractor import CourseConceptExtractor

data_dirs = {
    'slides': 'datasets/course-slides'
}

CourseConceptExtractor(data_dirs, debug=True)