# Set up spaCy
from spacy.en import English
parser = English()

# Test Data
sent = "There is an art, it says, or rather, a knack to flying." \
                 "The knack lies in learning how to throw yourself at the ground and miss." \
                 "In the beginning the Universe was created. This has made a lot of people "\
                 "very angry and been widely regarded as a bad move."
print len(sent)
multiSentence = sent.decode('utf-8')
# all you have to do to parse text is this:
#note: the first time you run spaCy in a file it takes a little while to load up its modules
parsedData = parser(multiSentence)

# Let's look at the tokens
# All you have to do is iterate through the parsedData
# Each token is an object with lots of different properties
# A property with an underscore at the end returns the string representation
# while a property without the underscore returns an index (int) into spaCy's vocabulary
# The probability estimate is based on counts from a 3 billion word
# corpus, smoothed using the Simple Good-Turing method.
for i, token in enumerate(parsedData):
    print("original:", token.orth, token.orth_)
    print("lowercased:", token.lower, token.lower_)
    print("lemma:", token.lemma, token.lemma_)
    print("shape:", token.shape, token.shape_)
    print("prefix:", token.prefix, token.prefix_)
    print("suffix:", token.suffix, token.suffix_)
    print("log probability:", token.prob)
    print("Brown cluster id:", token.cluster)
    print("----------------------------------------")
    if i > 1:
        break