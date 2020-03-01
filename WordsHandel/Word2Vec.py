from keras.layers import *
from gensim.models import Word2Vec

class BuildWord2Vec:
    '''
    Word 2 Vec
    '''

    def __init__(self, file_name):
        self.W2Vembedding = self.getEmbedding(file_name)

    def getEmbedding(self, file_name):
        W2Vembedding = Word2Vec.load(file_name)
        vocab_list = [word for word, Vocab in W2Vembedding.wv.vocab.items()]  # 存储 所有的 词语

        word_index = {}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
        word_vector = {}  # 初始化`[word : vector]`字典

        embeddings_matrix = np.zeros((len(vocab_list) + 1, W2Vembedding.vector_size))
        ## 3 填充 上述 的字典 和 大矩阵
        for i in range(len(vocab_list)):
            # 每个词语
            word = vocab_list[i]
            # 词语：序号
            word_index[int(word)] = i + 1
            # 词语：词向量
            word_vector[int(word)] = W2Vembedding.wv[word]
            # 词向量矩阵
            embeddings_matrix[i + 1] = W2Vembedding.wv[word]
        return embeddings_matrix


