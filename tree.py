import ppdb_utils

class tree(object):

    def __init__(self, phrase, words, pos_vocab):
        self.phrase = phrase
	self.phrase = [i.split('_')[0] for i in phrase.split() if len(i.split('_'))==2]
        self.pos = [i.split('_')[1] for i in phrase.split() if len(i.split('_'))==2]
        self.embeddings = []
	self.pos_embeddings = []
        self.representation = None

    def populate_embeddings(self, words, pos_vocab):
	for i in self.phrase:
            self.embeddings.append(ppdb_utils.lookupIDX(words,i))
	for i in self.pos:
	    self.pos_embeddings.append(ppdb_utils.lookupIDX(pos_vocab,i))

    def unpopulate_embeddings(self):
        self.embeddings = []
	self.pos_embeddings = []
