#!python

import nltk
import pprint
import sys

if __name__ == '__main__':
    """Generate templates to be used with Haikuify.haiku()"""

    haikus_txt = sys.argv[1]
    if not haikus_txt:
        print "usage: {} <haiku text file>".format(sys.argv[0])
        sys.exit(0)

    # assumess haikus in haikus_txt are separated by a new line
    haikus = []
    with open(haikus_txt, "r") as fd:
        haiku = []
        for line in fd:
            if line[-1] == "\n": line = line[0:-1]

            if line:
                haiku.append(line)
            else:
                haikus.append(haiku)
                haiku = []

    templates = []
    for h in haikus:
        template = []
        for l in h:
            template_line = []
            for t in nltk.word_tokenize(l):
                pos = nltk.pos_tag([t])[0][1]
                template_line.append(pos)
            template.append(tuple(template_line))
        templates.append(tuple(template))
        template = []

    pprint.PrettyPrinter(indent=4).pprint(sorted(templates))
