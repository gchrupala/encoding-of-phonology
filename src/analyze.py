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
    commands.add_parser('abx').set_defaults(func=abx)
    commands.add_parser('clustering').set_defaults(func=clustering)
    commands.add_parser('synonyms').set_defaults(func=synonyms)

    args = parser.parse_args()
    args.func(args)


def decoding(args):
    logging.info("Decoding")
    from phoneme_decode import decode
    decode()
    logging.info("Bootstrapping")
    from bootstrap import bootstrap
    bootstrap()


def abx(args):
    raise NotImplementedError

def clustering(args):
    raise NotImplementedError

def synonyms(args):
    from activations import activations, audio
    texts = [ line.strip() for line in open("synonym_sentences.txt")]
    audios = audio(texts, "../data/coco/synonym/")
    result = activations(audios, "../models/coco-speech.zip")
    numpy.save("mfcc.npy", result['mfcc'])
    numpy.save("states.npy", result['layer_states'])
    numpy.save("embeddings.npy", result['embeddings'])
    numpy.save("conv_states.npy", result['conv_states'])
    from synonyms import synonyms
    synonyms()


if __name__ == '__main__':
    main()
