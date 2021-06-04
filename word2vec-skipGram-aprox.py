from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

model = KeyedVectors.load_word2vec_format('word2vec-skipGram_s300.txt')

wordf = "transcricao.txt"

num_outputs = 15


def filter_results(outputs):
    newoutputs = []
    outputs.sort(key=lambda x: -x[1])
    occs = {w: 0 for w in [o[0][n] for n in range(4) for o in outputs]}
    toDelete = len(outputs) - num_outputs
    for o in outputs:
        for n in range(4):
            occs[o[0][n]] += 1
        uniqueResult = all([occs[o[0][n]] <= 4 for n in range(4)])
        if uniqueResult or toDelete == 0:
            newoutputs.append(o)
        else:
            toDelete -= 1
    return newoutputs


def approx_linear(model, words):
    outputs = []
    for i, first in enumerate(words):
        for j, second in enumerate(words[i+1:]):
            for third in words[(i+1)+(j+1):]:
                result = model.most_similar_cosmul(
                    positive=[first, second], negative=[third], topn=3)
                outputs += ([([first, second, third, result[n][0]],
                              result[n][1]) for n in range(3)])
        outputs = filter_results(outputs)
    outputs = outputs[:num_outputs]
    for o in outputs:
        print(o[0][0], "+", o[0][1], "-", o[0][2],
              "=", o[0][3]+" :", round(o[1], 3))


if __name__ == '__main__':

    w = open(wordf, "r")
    words = ",".join(w.read().split("\n"))
    w.close()

    words = [' '.join(word_tokenize(x)) for x in words.split(",")]

    words = [f for f in filter(lambda x: x in model.vocab.keys(), words)]

    approx_linear(model, words)