import gensim.downloader as api
import pandas as pd
import numpy as np
import csv


def evaluate_model(model_name:str):
    model = api.load(model_name)
    synonyms_file = pd.read_csv('synonyms.csv')
    synonyms = synonyms_file.to_numpy()
    output_file=open(f'{model_name}-details.csv','w',newline='')
    writer=csv.writer(output_file)
    writer.writerow(['Question','Correct-answer','Guess-word','Correctness'])
    for synonym in synonyms:
        if synonym[0] in model.key_to_index:
            similarity = model.most_similar(synonym[0])
            # for testing only
            print(similarity)

            guess=similarity[0][0]
            correctness= ''
            if synonym[1]==guess:
                correctness='correct'
            else:
                if guess in[synonym[2],synonym[3],synonym[4],synonym[5]]:
                    correctness='wrong'
                else:
                    correctness='guess'
            writer.writerow([synonym[0], synonym[1], guess, correctness])
        else:
            writer.writerow([synonym[0], synonym[1],"NA","guess"])









def main():
    evaluate_model("word2vec-google-news-300")


if __name__ == '__main__':
    main()