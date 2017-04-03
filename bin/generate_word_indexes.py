import nltk
import pickle

CMU_DICT = nltk.corpus.cmudict.dict()


def get_syllables(w):
    """Return the number of syllables in a word."""
    word = w.lower()
    if word not in CMU_DICT: raise KeyError
    return len([
        syllable
        for syllables in CMU_DICT[word]
        for syllable in syllables if syllable[-1].isdigit()])

def get_pos(w):
    """Return the part of speech of a word."""
    return nltk.pos_tag([w])[0][1]

def gen_word_indexes():
    """Generate indexes for words; pos_syllable_words and
    word_syllable_pos."""
    pos_syllable_words = {}  # syllable count -> part of speech -> words
    word_syllable_pos = {}  # word -> (part of speech, syllables)
    for word in CMU_DICT:
        pos = get_pos(word)
        syllables = get_syllables(word)
        pos_syllable_words.setdefault(pos, {})
        pos_syllable_words[pos].setdefault(syllables, []).append(word)
        word_syllable_pos[word] = (syllables, pos)
    return pos_syllable_words, word_syllable_pos

if __name__ == "__main__":
    print "generating indexes ..."
    pos_syllable_words, word_syllable_pos = gen_word_indexes()

    print "writing indexes to word_indexes.p ..."
    pickle.dump({
        "pos_syllable_words": pos_syllable_words,
        "word_syllable_pos": word_syllable_pos},
        open("word_indexes.p", "wb"))

    print "done."
