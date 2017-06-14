import imaginet.tts as tts
import logging
import requests
import time

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

