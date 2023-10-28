
from collections import Counter

def get_n_grams(n, file):
    tokens = []
    values = []
    count = Counter()
    for content in file:
        content = content[:-1] #removing new line character
        content = str.lower(content)
        content_tokens = content.split()
        tokens = tokens + content_tokens
    count = len(tokens) + 1 - n
    if n == 1:
        values = tokens
    else:
        for i in range(count):
            value = tuple()
            for j in range(n):
                value = value + (tokens[i + j],)
            values = values + [value,]
    return values, count

class NGrams():
    values = []
    count = Counter() 

    def __init__(self, n, file):
        file.seek(0)
        self.values, self.count = get_n_grams(n, file)
