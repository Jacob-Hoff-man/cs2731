
from collections import Counter

def get_n_grams(n, file, mode):
    tokens = []
    values = []
    for content in file:
        content = content[:-1] #removing new line character
        content = str.lower(content)
        content = '<s> ' + content + ' </s>'
        if mode == 'char':
            # chars
            tokens = tokens + list(content)
        else:
            # words
            content_tokens = content.split()
            tokens = tokens + content_tokens
    vocab = list(set(tokens))
    size = len(tokens) + 1 - n
    if n == 1:
        values = tokens
    else:
        for i in range(size):
            value = tuple(tokens[i : i + n])
            values = values + [value,]
    return values, vocab, size, Counter(values)

class NGrams():
    values = []
    vocab = []
    size = 0
    counts = Counter()
    token_type='char'
    def __init__(self, n, file, token_type='char'):
        file.seek(0)
        self.values, self.vocab, self.size, self.counts = get_n_grams(n, file, token_type)
