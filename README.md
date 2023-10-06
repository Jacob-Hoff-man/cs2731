# CS2710

## HW 1

- create a virtual enviornment using the `requirements.txt` file.
```
    python -m venv nlp-env
    source nlp-env/bin/activate
    pip install -r requirements.txt
    
```

- from inside the `/cs2731/hw1/` directory, run the script with the following command:

```
    python3 skeleton.py
```

- to perform the part 1 operations, please modify the `__main__` function:

```
    if __name__ == "__main__":
        words_part_1 = [...]
        words_part_2 = [...]
        identity_labels = [...]
        part_1(words_part_1)
        # part_2(words_part_2)
        # part_2(identity_labels)
        
```

- to perform the part 2 operations, please modify the `__main__` function:

```
    if __name__ == "__main__":
        words_part_1 = [...]
        words_part_2 = [...]
        identity_labels = [...]
        # part_1(words_part_1)
        part_2(words_part_2)
        part_2(identity_labels)
```

## HW 2

- copy the held-out test dataset into the `/cs2731/datasets/` directory.

- create a virtual environment using the `requirements.txt` file.

```
    python -m venv nlp-env
    source nlp-env/bin/activate
    pip install -r requirements.txt
    
```

- from inside the `/cs2731/` directory, run the following commands:

```
    # download spacy pipeline for pre-processing
    python -m spacy download en_core_web_sm
    # run script
    python3 hw2_jah292_test.py {held-out dataset file name}.csv
```

