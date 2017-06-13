from audio import save_audio
import logging
from itertools import izip
import hashlib

def encode(s):
    return hashlib.md5(s).hexdigest()

def main():
    logging.getLogger().setLevel('INFO')
    syls = [ line.split()[2] for line in open("abx_cv.txt") if not line.startswith("#") ][1:]
    save_audio(syls, "../data/coco/abx_cv/")
    sents = [ line.strip() for line in open("synonym_sentences.txt")]
    #audios = (open("../data/coco/synonym_iter/{}.wav".format(i), 'rb').read() for i in range(len(sents)))
    #just_save_audio(sents, audios, "../data/coco/synonym/")
    save_audio(sents, "../data/coco/synonym/")

def just_save_audio(texts, audios, audio_dir):
    logging.info("Storing wav files")
    for text, audio in izip(texts, audios):
        logging.info("Storing audio for {}".format(text))

        path = encode(text)
        with open("{}/{}.wav".format(audio_dir, path), 'w') as out:
            out.write(audio)

if __name__ == '__main__':
    main()
