import numpy
import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
from audio import extract_mfcc
import sys
import argparse
import logging
import audio


def activations(audios, model_path):
    """Return layer states and embeddings for sentences in audios,
    extracting MFCC features and applying a speech model to them.
    """
    logging.info("Loading model")
    model = task.load(model_path)

    logging.info("Extracting MFCC features")
    mfccs  = [ extract_mfcc(au) for au in audios]
    logging.info("Extracting convolutional states")
    conv_states = audiovis.conv_states(model, mfccs)
    logging.info("Extracting layer states")
    states = audiovis.layer_states(model, mfccs)
    logging.info("Extracting sentence embeddings")
    embeddings = audiovis.encode_sentences(model, mfccs)
    return {'mfcc': mfccs, 'conv_states': conv_states, 'layer_states': states, 'embeddings': embeddings}

def save_activations(audios, model_path, mfcc_path, conv_path, states_path, emb_path):
    """Return layer states and embeddings for sentences in audios,
    extracting MFCC features and applying a speech model to them.
    """
    logging.info("Loading model")
    model = task.load(model_path)
    audios = list(audios)
    logging.info("Extracting MFCC features")
    mfccs  = [ extract_mfcc(au) for au in audios]
    numpy.save(mfcc_path, mfccs)
    logging.info("Extracting convolutional states")
    numpy.save(conv_path, audiovis.conv_states(model, mfccs))
    logging.info("Extracting layer states")
    numpy.save(states_path, audiovis.layer_states(model, mfccs))
    logging.info("Extracting sentence embeddings")
    numpy.save(emb_path, audiovis.encode_sentences(model, mfccs))

def main():
    logging.getLogger().setLevel('INFO')
    model_path="../models/coco-speech.zip"
    parser = argparse.ArgumentParser()
    parser.add_argument('texts',
                            help='Path to file with texts')
    parser.add_argument('--model', default=model_path,
                            help='Path to file with model')
    parser.add_argument('--mfcc', default='mfcc.npy',
                            help='Path to file where MFCCs will be stored')
    parser.add_argument('--layer_states', default='states.npy',
                            help='Path to file where layer states will be stored')
    parser.add_argument('--conv_states', default='conv_states.npy',
                            help='Path to file where state of convolutional layer will be stored')
    parser.add_argument('--embeddings', default='embeddings.npy',
                            help='Path to file where sentence embeddings will be stored')
    parser.add_argument('--audio_dir', default="/tmp",
                            help='Path to directory where audio is stored')
    parser.add_argument('--synthesize', action='store_true', default=False,
                            help='Should audio be synthesized')
    args = parser.parse_args()
    texts = [ line.strip() for line in open(args.texts)]
    if args.synthesize:
        audio.save_audio(texts, args.audio_dir)
    audios = audio.load_audio(texts, args.audio_dir)
    save_activations(audios, args.model,
        args.mfcc, args.conv_states, args.layer_states, args.embeddings)


if __name__=='__main__':
    main()
