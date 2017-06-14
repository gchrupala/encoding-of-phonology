
import logging
import python_speech_features as features
import scipy.io.wavfile as wav
import StringIO
import numpy
import hashlib


def extract_mfcc(sound):
    (rate,sig) = wav.read(StringIO.StringIO(sound))
    mfcc_feat = features.mfcc(sig,rate)
    return numpy.asarray(mfcc_feat, dtype='float32')

def encode(s):
    return hashlib.md5(s).hexdigest()

def load_audio(texts, audio_dir):
    """Load audio from audio_dir.
    """
    logging.info("Loading audio")
    for text in texts:
        path = encode(text)
        with open("{}/{}.wav".format(audio_dir, path), "rb") as au:
            yield au.read()
