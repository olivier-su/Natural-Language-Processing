import gensim.downloader as api
import pandas as pd
import numpy as np
import csv


def evaluate_model(model_name: str):
    model = api.load(model_name)
    synonyms_file = pd.read_csv('synonyms.csv')
    synonyms = synonyms_file.to_numpy()
    output_file = open(f'{model_name}-details.csv', 'w', newline='')
    detail_writer = csv.writer(output_file)
    detail_writer.writerow(['Question', 'Correct-answer', 'Guess-word', 'Correctness'])
    for synonym in synonyms:
        if synonym[0] in model.key_to_index:

            options = [synonym[2], synonym[3], synonym[4], synonym[5]]
            (guess, score) = model.most_similar(synonym[0], topn=1)[0]
            question = synonym[0]
            if guess not in options:
                score = -1.0
                for i in options:
                    if i in model.key_to_index:
                        score_temp = model.similarity(question, i)
                        if score_temp > score:
                            guess = i
                            score = score_temp

            correctness = ''
            if synonym[1] == guess:
                correctness = 'correct'
            else:
                if guess in options:
                    correctness = 'wrong'
                else:
                    correctness = 'guess'
            detail_writer.writerow([synonym[0], synonym[1], guess, correctness])
        else:
            detail_writer.writerow([synonym[0], synonym[1], "NA", "guess"])


def main():
    evaluate_model("word2vec-google-news-300")


if __name__ == '__main__':
    main()
