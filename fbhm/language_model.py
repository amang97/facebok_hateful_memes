class LanguageModel:
    def __init__(self):
        super().__init__()

    def text_embedding(self, tdf, mode):
        """
        Self - Attention Model w/ context vector: Average (word embeddings)
        Sentence level embeddings

        Input:
        Word embedding, Average them to get sentence embedding call it 
        context vector. Use an MLP with input word embedding, context vector

        Output:
        outputs weights for each word embedding (softmax over the weights)

        Assumes:
        te is np.array
        """
        # TODO
        te = None
        return te