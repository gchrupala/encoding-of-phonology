import numpy
import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
import imaginet.tts as tts
import sys
import argparse
import logging
import os
import requests
import time
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def synthesize(text, path=None, trial=1):
    try:
        return tts.decodemp3(tts.speak(text))
    except requests.exceptions.HTTPError:
        if trial > 10:
            raise RuntimeError("HTTPError: giving up after 10 trials")
        else:
            logging.info("HTTPError, waiting for 5 sec")
            time.sleep(5)
            return synthesize(text, path=path, trial=trial+1)

def make_audio(texts, audio_dir):
    """Synthesize and store audio in audio_dir.
    """
    logging.info("Synthesizing speech")
    mkdir_p(audio_dir)
    audios = (synthesize(text) for text in texts)
    logging.info("Storing wav files")
    for i, audio in enumerate(audios):
        with open("{}/{}.wav".format(audio_dir, i), 'w') as out:
            out.write(audio)

def activations(audios, model_path):
    """Return layer states and embeddings for sentences in audios,
    extracting MFCC features and applying a speech model to them.
    """
    logging.info("Loading model")
    model = task.load(model_path)

    logging.info("Extracting MFCC features")
    mfccs  = [ tts.extract_mfcc(au) for au in audios]
    logging.info("Extracting convolutional states")
    conv_states = audiovis.conv_states(model, mfccs)
    logging.info("Extracting layer states")
    states = audiovis.layer_states(model, mfccs)
    logging.info("Extracting sentence embeddings")
    embeddings = audiovis.encode_sentences(model, mfccs)
    return {'mfcc': mfccs, 'conv_states': conv_states, 'layer_states': states, 'embeddings': embeddings}

def audio(texts, audio_dir):
    """Load audio from audio_dir.
    """
    logging.info("Loading audio")
    for i in range(len(texts)):
        with open("{}/{}.wav".format(audio_dir, i), "rb") as au:
            yield au.read()

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
        make_audio(texts, args.audio_dir)
    audios = audio(texts, args.audio_dir)
    result = activations(audios, args.model)
    numpy.save(args.mfcc, result['mfcc'])
    numpy.save(args.layer_states, result['layer_states'])
    numpy.save(args.embeddings, result['embeddings'])
    numpy.save(args.conv_states, result['conv_states'])


if __name__=='__main__':
    main()
