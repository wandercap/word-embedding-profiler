from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

model = KeyedVectors.load_word2vec_format('word2vec-skipGram_s300.txt')

wordf = "transcricao.txt"

axes = [0, 1, 2]

clusterK = 5

num_outputs = 15

colors = ["tab:red", "tab:blue", "tab:green", "tab:orange",
          "tab:purple", "tab:olive", "tab:pink", "tab:cyan", "tab:gray"]
defaultcolor = "black"

sizes = []
defaultsize = 16

def plot2D(result, wordgroups):
    pyplot.scatter(result[:, axes[0]], result[:, axes[1]])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            # Create plot point
            coord = (result[i, axes[0]], result[i, axes[1]])
            color = colors[g] if g < len(colors) else defaultcolor
            size = sizes[g] if g < len(sizes) else defaultsize
            pyplot.annotate(word, xy=coord, color=color, fontsize=size)


def plot3D(result, wordgroups):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, axes[0]], result[:, axes[1]], result[:, axes[2]])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            # Create plot point
            color = colors[g] if g < len(colors) else defaultcolor
            size = sizes[g] if g < len(sizes) else defaultsize
            ax.text(result[i, axes[0]], result[i, axes[1]],
                    result[i, axes[2]], word, color=color, fontsize=size)


def get_groups(wordf, model):
    words = []
    groups = []

    # Extract words to plot from file
    for line in open(wordf, "r", encoding="utf-8").read().split("\n"):
        l = [' '.join(word_tokenize(x)) for x in line.split(",")]
        l = filter(lambda x: x in model.vocab.keys(), l)
        groups.append(l)
        words += l

    # Get word vectors from model
    vecs = {w: model.vocab[w] for w in words}

    # Assign groups if using clustering
    if clusterK > 0:
        estimator = KMeans(init='k-means++', n_clusters=clusterK, n_init=10)
        estimator.fit_predict(model[vecs])
        groups = [[] for n in range(clusterK)]
        for i, w in enumerate(vecs.keys()):
            group = estimator.labels_[i]
            groups[group].append(w)

    return words, groups, vecs


if __name__ == '__main__':

    # Get groups from file or by clustering
    words, groups, vecs = get_groups(wordf, model)

    coords = model[vecs]

    # Create axes to plot on
    pca = PCA(n_components=max(axes)+1)
    result = pca.fit_transform(coords)

    # Plot vectors on axes
    if len(axes) > 2:
        plot3D(result, groups)
    else:
        plot2D(result, groups)
    pyplot.show()