# CS2710

## HW 2 - Text Classification

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
