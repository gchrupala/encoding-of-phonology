
import imaginet.tts as tts
import base64
import logging
import os
import errno
import requests
import time
from itertools import izip
import hashlib

def encode(s):
    return hashlib.md5(s).hexdigest()

def synthesize(text, path=None, trial=1):
    logging.info("Synthesizing {}".format(text))
    try:
        return tts.decodemp3(tts.speak(text))
    except requests.exceptions.HTTPError:
        if trial > 10:
            raise RuntimeError("HTTPError: giving up after 10 trials")
        else:
            logging.info("HTTPError on trial {}, waiting for 5 sec".format(trial))
            time.sleep(5)
            return synthesize(text, path=path, trial=trial+1)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_audio(texts, audio_dir):
    """Synthesize and store audio in audio_dir.
    """
    logging.info("Synthesizing speech")
    mkdir_p(audio_dir)
    audios = (synthesize(text) for text in texts)
    logging.info("Storing wav files")
    for text, audio in izip(texts, audios):
        logging.info("Storing audio for {}".format(text))

        path = encode(text)
        with open("{}/{}.wav".format(audio_dir, path), 'w') as out:
            out.write(audio)

def load_audio(texts, audio_dir):
    """Load audio from audio_dir.
    """
    logging.info("Loading audio")
    for text in texts:
        path = encode(text)
        with open("{}/{}.wav".format(audio_dir, path), "rb") as au:
            yield au.read()
