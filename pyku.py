import nltk
import pickle
import random


word_indexes = pickle.load(open("word_indexes.p", "rb"))
pos_syllable_words = word_indexes["pos_syllable_words"]
word_syllable_pos = word_indexes["word_syllable_pos"]


POS_TOKENS = set([
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS",
    "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP",
    "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP",
    "WP$"])

TEMPLATES = (
    (   ('DT', 'JJ', 'NN'),
        ('NNS', 'PRP', 'IN', 'DT', 'NN'),
        ('IN', 'DT', 'NN', 'NN', 'NN')),
    (   ('DT', 'NN', 'NN', 'NNS', '.'),
        ('CC', 'DT', 'NN', 'NNS', 'VBG'),
        ('IN', 'IN', 'DT', 'NNS')),
    (   ('DT', 'NN', 'NN', 'VBZ', 'NN', ':'),
        ('NNS', 'NN', 'RB', 'TO', 'PRP$', 'NN', ','),
        ('VBN', 'NN', 'JJ')),
    (   ('DT', 'RB', 'NN', 'NN'),
        ('RB', 'NNS', 'IN', 'PRP'),
        ('VBN', 'NN', 'NNS')),
    (   ('IN', 'DT', 'NNS'),
        ('NNS', 'NN', 'NNS', 'IN', 'NN'),
        ('NN', 'DT', 'NN', 'NN')),
    (   ('IN', 'NNS', 'VBG', ','),
        ('PRP', 'NN', 'NN', 'PRP', 'MD', 'NN'),
        ('DT', 'VBG', 'NN')),
    (   ('JJ', 'IN', 'PRP$', 'NN', ','),
        ('RB', 'PRP$', 'NNS', 'MD', 'NN'),
        ('DT', 'NN', 'NNS')),
    (('NN', 'CC', 'NNS'), ('VBG', 'NN'), ('DT', 'JJ', 'NN')),
    (   ('NN', 'IN', 'NN', 'NN'),
        ('CC', 'VBN', 'IN', 'NN', ',', 'DT', 'NN'),
        ('NNS', 'RB', 'NN')),
    (   ('NN', 'IN', 'NN', 'NNS'),
        ('IN', 'DT', 'NN', 'NN', 'NN'),
        ('DT', 'JJ', 'VBG', 'NN')),
    (   ('NN', 'NN', 'IN', 'PRP$'),
        ('NN', 'IN', 'NN', ','),
        ('VBG', 'PRP$', 'NN', 'RB')),
    (('NN', 'NNS'), ('DT', 'IN', 'NNS'), ('IN', 'NNS', 'NNS')),
    (   ('NN', 'NNS', 'NN', 'IN', '.'),
        ('DT', 'NN', 'NNS', 'NN', '.'),
        ('DT', 'NN', 'VBG', '.')),
    (   ('NN', 'VBZ', 'VBG'),
        ('RB', 'IN', 'DT', 'NN', 'CC', 'PRP', ','),
        ('DT', 'NN', 'VBG')),
    (   ('PRP', 'VBD', 'IN', 'DT', 'NNS'),
        ('TO', 'NN', ':'),
        ('PRP', 'VBD', 'RB', 'NN')),
    (('RB', ',', 'IN', 'JJ'), ('NNS', ',', 'VBG', 'NN'), ('NN', 'DT', 'NN')),
    (   ('RB', 'NN', 'NN', 'NN', ','),
        ('CD', 'JJ', 'NNS'),
        ('VBG', 'IN', 'PRP$', 'NN')),
    (('VBG', 'NN', ':'), ('DT', 'NN', 'NNS', ','), ('CD', 'VB', 'VBN')),
    (   ('VBG', 'NN', ':'),
        ('VBG', 'PRP$', 'NN', 'IN', 'NNS'),
        ('TO', 'DT', 'NN')),
    (   ('VBG', 'NN', 'NNS', ','),
        ('NN', 'IN', 'NN', 'NN', 'NNS', ','),
        ('DT', 'NN', 'NNS')),
    (   ('VBG', 'NNS', 'IN'),
        ('NN', 'NNS', ',', 'IN', 'CD', 'NN'),
        ('PRP', 'NNS', 'PRP$', 'NN')))

#----------------------------------------------------------------------------
# TODO move to utils package

def get_synonyms(w):
    '''Return a list of synonyms for a given word.'''
    synonyms = set()
    for syn in nltk.corpus.wordnet.synsets(w):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def get_pos(w):
	"""Return part of speech for a word."""
	return nltk.pos_tag([w])[0][1]


def get_syllables(w):
    """Return the number of syllables in a word."""
    word = w.lower()
    if word not in CMU_DICT: raise KeyError
    return len([
        syllable
        for syllables in CMU_DICT[word]
        for syllable in syllables if syllable[-1].isdigit()])

#----------------------------------------------------------------------------

def _get_max_next_syllable(current_token, template_len, syllables_left):
    is_last_token = current_token == template_len
    tokens_left = template_len - current_token
    return syllables_left \
        if is_last_token \
        else syllables_left - tokens_left


def _execute_haiku_line_template(template,
                                 max_syllables=5,
                                 subject=None,           # word to add;
                                                         # depending on word_emphasis
                                 subject_emphasis=0.0):  # % of adding word
    """Return a haiku line based on the template,
    throws LookupError if line cannot be generated."""
    syllables_left = max_syllables

    # syllables are based on POS tokens left in template
    template_len = len([t for t in template if t in POS_TOKENS])
    assert template_len <= max_syllables, "more word tokens than allowed syllables"

    subject_synonyms = {}
    if subject:
		# TODO handle compound synonyms
        for s in get_synonyms(subject):
			if s not in word_syllable_pos:
				continue
			syllables, pos = word_syllable_pos[s]
			subject_synonyms.setdefault(pos, {})
			subject_synonyms[pos].setdefault(syllables, []).append(s)

    subject_used = False
    executed = []
    for i, pos in enumerate(template, start=1):
        word = ""
        if pos not in POS_TOKENS:
            # not a part of speech, e.g. punctuation, so use as literal
            word = pos
        else:
            syllable_words = pos_syllable_words[pos]
            random_syllable_count = random.randint(
                1, _get_max_next_syllable(i, template_len, syllables_left))
            if subject_synonyms \
              and not subject_used \
			  and pos in subject_synonyms \
              and random_syllable_count in subject_synonyms[pos] \
              and random.random() < subject_emphasis:
                syllable_words = subject_synonyms[pos]
                subject_used = True

            if random_syllable_count not in syllable_words:
                raise LookupError
            word = random.choice(syllable_words[random_syllable_count])

        executed.append(word)
        if word not in word_syllable_pos:
            # TODO log to INFO likely punctuation
            continue
        syllables_left = syllables_left - word_syllable_pos[word][0]

    return " ".join(executed)


def _get_haiku_syllables_by_line_number(i):
    """Returns number of syllables for 5-7-5 structured haikus based on line
    number, starting with 0 as the first line."""
    return 5 if i % 2 == 0 else 7


def _get_templates_by_syllables_and_pos(syllables, pos, templates=TEMPLATES):
    """Returns filtered template list based on part of speech and if syllable
    count of pos does not exceed target syllables for line; assumes 5-7-5 haiku structure"""
    filtered = []
    for t in templates:
        match = False
        for i, line in enumerate(t):
            if match: break
            syllables_and_all_words  = syllables + len(line) - 1
            max_syllables = _get_haiku_syllables_by_line_number(i)
            for l_pos in line:
                if l_pos == pos and syllables_and_all_words <= max_syllables:
                    filtered.append(t)
                    match = True
                    break
    return filtered


def haiku(templates=TEMPLATES, subject=None, max_retries=10):
    """Return a list composed of three haiku lines."""

    subject_syllable_pos = word_syllable_pos[subject] \
            if subject in word_syllable_pos \
            else None

    templates = _get_templates_by_syllables_and_pos(*subject_syllable_pos) \
            if subject_syllable_pos \
            else templates
    assert 0 < len(templates), "no templates available"

    tries = 0
    while tries < max_retries:
    	template = random.choice(templates)
    	haiku = []
        try:
			for i, line in enumerate(template):
				# TODO vary emphasis depending on if subject used
				haiku.append(_execute_haiku_line_template(
					line,
					max_syllables=_get_haiku_syllables_by_line_number(i),
					subject=subject,
					subject_emphasis=0.20))
			return haiku
        except LookupError:
			pass
        tries += 1

if __name__ == '__main__':
    for s in ('porkupines', 'panda', 'river', 'egg'):
        print "subject={}".format(s)
        print haiku(subject=s)
