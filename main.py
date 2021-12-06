import gensim.downloader as api
import pandas as pd
import numpy as np
import csv


def evaluate_model(model_name: str, analysis_writer):

    print(f'Starting model {model_name}')

    model = api.load(model_name)
    synonyms_file = pd.read_csv('synonyms.csv')
    synonyms = synonyms_file.to_numpy()
    output_file = open(f'{model_name}-details.csv', 'w', newline='')
    detail_writer = csv.writer(output_file)
    detail_writer.writerow(['Question', 'Correct-answer', 'Guess-word', 'Correctness'])
    correct_count = 0
    question_count = 0
    for synonym in synonyms:

        if synonym[0] in model.key_to_index:

            options = [synonym[2], synonym[3], synonym[4], synonym[5]]
            (guess, score) = model.most_similar(synonym[0], topn=1)[0]
            question = synonym[0]
            if guess not in options:
                score = -1.0
                for option in options:
                    if option in model.key_to_index:
                        score_temp = model.similarity(question, option)
                        if score_temp > score:
                            guess = option
                            score = score_temp

            correctness = ''
            if synonym[1] == guess:
                correctness = 'correct'
                correct_count += 1
                question_count += 1
            else:
                if guess in options:
                    correctness = 'wrong'
                    question_count += 1
                else:
                    correctness = 'guess'
            detail_writer.writerow([synonym[0], synonym[1], guess, correctness])
        else:
            detail_writer.writerow([synonym[0], synonym[1], "NA", "guess"])
    output_file.close()
    analysis_writer.writerow(
        [model_name, len(model.key_to_index), correct_count, question_count, correct_count / question_count])


def main():
    analysis_csv = 'analysis.csv'
    analysis_file = open(analysis_csv, 'w', newline='')
    analysis_writer = csv.writer(analysis_file)
    analysis_writer.writerow(
        ['Model Name', 'Size Of Vocabulary', 'C', 'V', 'Accuracy'])

    evaluate_model("word2vec-google-news-300", analysis_writer)
    evaluate_model("fasttext-wiki-news-subwords-300", analysis_writer)
    evaluate_model("glove-wiki-gigaword-300", analysis_writer)
    evaluate_model("glove-twitter-25", analysis_writer)
    evaluate_model("glove-twitter-200", analysis_writer)

    analysis_file.close()


if __name__ == '__main__':
    main()
