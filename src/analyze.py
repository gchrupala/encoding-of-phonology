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

def abx_all(args):
    logging.info("ABX task - all syllables")
    import abx
    abx.abx_all()

def abx_classes(args):
    logging.info("ABX task - divided by phoneme class")
    import abx
    abx.abx_classes()
    abx.abx_cv_scores()

def clustering(args):
    raise NotImplementedError

def synonyms(args):
    from activations import save_activations
    from audio import load_audio
    texts = [ line.strip() for line in open("synonym_sentences.txt")]
    audios = load_audio(texts, "../data/coco/synonym/")
    save_activations(audios, "../models/coco-speech.zip",
        "mfcc.npy", "conv_states.npy", "states.npy", "embeddings.npy")
    from synonyms import synonyms
    synonyms()


if __name__ == '__main__':
    main()
