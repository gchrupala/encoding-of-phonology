from __future__ import division
from __future__ import print_function
import numpy
import argparse
import logging

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    commands.add_parser('decoding').set_defaults(func=decoding)
    commands.add_parser('abx_all').set_defaults(func=abx_all)
    commands.add_parser('abx_classes').set_defaults(func=abx_classes)
    commands.add_parser('clustering').set_defaults(func=clustering)
    commands.add_parser('correlation').set_defaults(func=correlation)
    commands.add_parser('synonyms').set_defaults(func=synonyms)
    commands.add_parser('dendro').set_defaults(func=dendro)

    args = parser.parse_args()
    args.func(args)


def decoding(args):
    logging.info("Decoding")
    from phoneme_decode import decode
    decode()
    logging.info("Bootstrapping")
    from bootstrap import bootstrap
    bootstrap()

def abx_all(args):
    logging.info("ABX task - all syllables")
    import abx
    abx.abx_all()

def abx_classes(args):
    logging.info("ABX task - divided by phoneme class")
    import abx
    abx.abx_classes()
    abx.abx_cv_scores()

def correlation(args):
    logging.info("Phoneme correlation")
    import clustering
    clustering.phoneme_correlation()

def clustering(args):
    logging.info("Phoneme clustering")
    import clustering
    clustering.phoneme_clustering()

def synonyms(args):
    from activations import save_activations
    from audio import load_audio
    texts = [ line.strip() for line in open("synonym_sentences.txt")]
    audios = load_audio(texts, "../data/coco/synonym/")
    save_activations(audios, "../models/coco-speech.zip",
        "mfcc.npy", "conv_states.npy", "states.npy", "embeddings.npy")
    from synonyms import synonyms
    synonyms()

def dendro(args):
    import cPickle as pickle
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    from sklearn.cluster import AgglomerativeClustering
    phones = pickle.load(open("phoneme_cluster_keys.pkl"))
    phoneme_cluster = pickle.load(open("phoneme_cluster_rec0.pkl"))
    lines = list(open('phonemes.txt'))[1:]
    ipa = {}
    for line in lines:
        cols = line.strip().split('\t')
        ipa[cols[0]] = cols[1].decode('utf8')
    labels = [ipa[phones[x]] for x in range(len(phones))]
    plt.figure(figsize=(16,8))
    plt.rc('font', family='DejaVu Sans')
    plt.yticks([])
    plot_dendrogram(phoneme_cluster, labels=labels, leaf_font_size=16, truncate_mode=None, show_contracted=False,
                link_color_func=lambda k: 'black')
    plt.savefig('../figures/dendro.pdf')

def plot_dendrogram(model, **kwargs):
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, color_threshold=0, **kwargs)


if __name__ == '__main__':
    main()
