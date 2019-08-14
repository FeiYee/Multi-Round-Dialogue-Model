from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras_layer_normalization import LayerNormalization

from gensim.models import Word2Vec

class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """

    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape) - 1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs

class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                layer.build(K.int_shape(args[0]))
            else:
                layer.build(K.int_shape(kwargs['inputs']))
            self._trainable_weights.extend(layer._trainable_weights)
            self._non_trainable_weights.extend(layer._non_trainable_weights)
        return layer.call(*args, **kwargs)

class SelfModulatedLayerNormalization(OurLayer):
    """模仿Self-Modulated Batch Normalization，
    只不过降Batch Normalization改为Layer Normalization
    """
    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden
    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        output_dim = input_shape[0][-1]
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = Dense(output_dim)
        self.gamma_dense_1 = Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = Dense(output_dim)

    def call(self, inputs):
        inputs, cond = inputs
        inputs = self.reuse(self.layernorm, inputs)
        beta = self.reuse(self.beta_dense_1, cond)
        beta = self.reuse(self.beta_dense_2, beta)
        gamma = self.reuse(self.gamma_dense_1, cond)
        gamma = self.reuse(self.gamma_dense_2, gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return inputs * (gamma + 1) + beta
    def compute_output_shape(self, input_shape):
        return input_shape[0]

class Attention(OurLayer):
    """多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

class OurBidirectional(OurLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = copy.deepcopy(layer)
        self.backward_layer = copy.deepcopy(layer)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], 2)
        return x * mask
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.forward_layer.units * 2)


class BuildModel:
    def __init__(self,chars_num = None, w2v_file = None):
        self.batch_size = 32
        self.learning_rate = 0.0008
        self.max_len = 64
        self.char_size = 256
        self.z_dim = 128
        self.chars_num = chars_num

        if w2v_file == None:
            self.word2vec = None
            if chars_num == None:
                print("chars_num 或 w2v_file 必须至少满足一个条件!")
                exit(0)
        else:
            self.word2vec = self.BuildWord2Vec(w2v_file)
        self.model = self.build()

    class BuildWord2Vec:
        ‘’‘
            Word2Vec转化
        ’‘’
        def __init__(self,file_name):
            self.W2Vembedding = self.getEmbedding(file_name)

        def getEmbedding(self,file_name):
            W2Vembedding = Word2Vec.load(file_name)
            vocab_list = [word for word, Vocab in W2Vembedding.wv.vocab.items()]  # 存储 所有的 词语

            word_index = {}  # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。
            word_vector = {}  # 初始化`[word : vector]`字典

            embeddings_matrix = np.zeros((len(vocab_list) + 1, W2Vembedding.vector_size))
            ## 3 填充 上述 的字典 和 大矩阵
            for i in range(len(vocab_list)):
                # print(i)
                word = vocab_list[i]  # 每个词语
                word_index[int(word)] = i + 1  # 词语：序号
                word_vector[int(word)] = W2Vembedding.wv[word]  # 词语：词向量
                embeddings_matrix[i + 1] = W2Vembedding.wv[word]  # 词向量矩阵
            return embeddings_matrix
    def seq_maxpool(self,x):
        """seq是[None, seq_len, s_size]的格式，
        mask是[None, seq_len, 1]的格式，先除去mask部分，
        然后再做maxpooling。
        """
        seq, mask = x
        seq -= (1 - mask) * 1e10
        return K.max(seq, 1)

    def build(self):
        '''
        根据数据集自行定义
        基础信息部分：session_info
            用户级别\用户年龄\品类\订单状态
            字段名称    取值范围    长度
            用户级别：   0 ~ 5       :6
            用户年龄：   0 ~ 7       :8
            商品品类：   0 ~ 2484    :2485
            订单状态：   0 ~ 3       :4
        '''
        
        session_info = Input(shape=(None,))
        '''
        标准信息部分：
            问题输入标量： x_in
            正向回答标量： yl_in
            逆向回答标量： yr_in
            历史问题标量： z_in
        '''
        x_in = Input(shape=(None,))
        yl_in = Input(shape=(None,))
        yr_in = Input(shape=(None,))
        z_in = Input(shape=(None,))
        x, yl,yr, z = x_in, yl_in, yr_in, z_in

        session_level = Lambda(lambda x: x[:,0])(session_info)
        session_years = Lambda(lambda x: x[:,1])(session_info)
        session_kinds = Lambda(lambda x: x[:,2])(session_info)
        session_station = Lambda(lambda x: x[:,3])(session_info)

        session_level = Embedding(6, self.char_size // 4)(session_level)
        session_years = Embedding(8, self.char_size // 4)(session_years)
        session_kinds = Embedding(2485, self.char_size // 4)(session_kinds)
        session_station = Embedding(4, self.char_size // 4)(session_station)

        session = Concatenate()([session_level, session_years, session_kinds, session_station])
        session = Lambda(lambda x: K.expand_dims(x, 1))(session)

        x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
        y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(yl)
        z_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(z)

        x_one_hot = Lambda(self.to_one_hot)([x, x_mask])
        z_one_hot = Lambda(self.to_one_hot)([z, z_mask])
        xz_one_hot = Lambda(lambda x: K.cast(K.greater(x[0] + x[1], 0.5), 'float32'))([x_one_hot, z_one_hot])
        xz_prior = ScaleShift()(xz_one_hot)  # 学习输出的先验分布

        if self.word2vec != None:
            embedding = Embedding(len(self.word2vec.W2Vembedding),  # 字典长度
                                    self.char_size,  # 词向量 长度（100）
                                    weights=[self.word2vec.W2Vembedding],  # 重点：预训练的词向量系数
                                    trainable=True  # 是否在 训练的过程中 更新词向量
                                    )
        else:
            embedding = Embedding(self.chars_num + 4, self.char_size)

        x = embedding(x)
        z = embedding(z)

        # encoder，双层双向LSTM
        x = LayerNormalization()(x)
        x = OurBidirectional(CuDNNLSTM(self.z_dim // 2, return_sequences=True))([x, x_mask])
        x = LayerNormalization()(x)
        x = OurBidirectional(CuDNNLSTM(self.z_dim // 2, return_sequences=True))([x, x_mask])

        z = LayerNormalization()(z)
        z = OurBidirectional(CuDNNLSTM(self.z_dim // 2, return_sequences=True))([z, z_mask])
        z = LayerNormalization()(z)
        z = OurBidirectional(CuDNNLSTM(self.z_dim // 2, return_sequences=True))([z, z_mask])

        x_max = Lambda(self.seq_maxpool)([x, x_mask])

        session = LayerNormalization()(session)
        session = CuDNNLSTM(self.z_dim // 4, return_sequences=True)(session)
        session = LayerNormalization()(session)
        session = CuDNNLSTM(self.z_dim // 4, return_sequences=True)(session)
        session = LayerNormalization()(session)
        session = CuDNNLSTM(self.z_dim // 4, return_sequences=True)(session)

        # 正向decoder，单向LSTM
        y = embedding(yl)
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = CuDNNLSTM(self.z_dim, return_sequences=True)(y)
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = CuDNNLSTM(self.z_dim, return_sequences=True)(y)
        yl = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])

        # 逆向decoder，单向LSTM
        y = embedding(yr)
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = CuDNNLSTM(self.z_dim, return_sequences=True)(y)
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = CuDNNLSTM(self.z_dim, return_sequences=True)(y)
        yr = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])

        # 对齐attention + 检索attention
        yl_ = Attention(8, 16, mask_right=True)([yl, yr, yr])
        ylx = Attention(8, 16)([yl, x, x, x_mask])
        ylz = Attention(8, 16)([yl, z, z, z_mask])
        yls = Attention(8, 16, mask_right=True)([yl, session, session])
        yl = Concatenate()([yl, yl_, ylx, ylz, yls])
        # 对齐attention + 检索attention
        yr_ = Attention(8, 16, mask_right=True)([yr, yl, yl])
        yrx = Attention(8, 16)([yr, x, x, x_mask])
        yrz = Attention(8, 16)([yr, z, z, z_mask])
        yrs = Attention(8, 16, mask_right=True)([yr, session, session])
        yr = Concatenate()([yr, yr_, yrx, yrz, yrs])

        # 最后的输出分类（左右共享权重）
        classifier = Dense(len(self.data_info.chars) + 4)
        yl = Dense(self.data_info.char_size)(yl)
        yl = LeakyReLU(0.2)(yl)
        yl = classifier(yl)
        yl = Lambda(lambda x: (x[0] + x[1]) / 2)([yl, xz_prior])  # 与先验结果平均
        yl = Activation('softmax')(yl)
        yr = Dense(self.data_info.char_size)(yr)
        yr = LeakyReLU(0.2)(yr)
        yr = classifier(yr)
        yr = Lambda(lambda x: (x[0] + x[1]) / 2)([yr, xz_prior])  # 与先验结果平均
        yr = Activation('softmax')(yr)

        # 交叉熵作为loss，但mask掉padding部分
        cross_entropy_1 = K.sparse_categorical_crossentropy(yl_in[:, 1:], yl[:, :-1])
        cross_entropy_1 = K.sum(cross_entropy_1 * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])
        cross_entropy_2 = K.sparse_categorical_crossentropy(yr_in[:, 1:], yr[:, :-1])
        cross_entropy_2 = K.sum(cross_entropy_2 * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])
        cross_entropy = (cross_entropy_1 + cross_entropy_2) / 2

        model = Model([session_info, x_in, yl_in, yr_in, z_in], [yl, yr])
        model.add_loss(cross_entropy)
        model.compile(optimizer = Adam(self.learning_rate))
        # print(model.summary())
        return model

    def gen_sent(self,s, session, h = '',topk = 3,
               maxlen_query=None,
               maxlen_history=None,
               maxlen_answer=None):
        if maxlen_query == None:
            maxlen_query = self.max_len
        if maxlen_history == None:
            maxlen_history = self.max_len * 2
        if maxlen_answer == None:
            maxlen_answer = self.max_len
        """双向beam search解码
        每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
        """
        xid = np.array([self.data_info.str2id(s,maxlen_query)] * topk ** 2)  # 输入转id
        yl_id = np.array([[2]] * topk)  # L2R解码均以<start>开头，这里<start>的id为2
        yr_id = np.array([[3]] * topk)  # R2L解码均以<end>开头，这里<end>的id为3
        zid = np.array([self.data_info.str2id(h[::-1], maxlen_history)] * topk ** 2) # 历史转id
        session = np.array([session] * topk ** 2 )
        l_scores, r_scores = [0] * topk, [0] * topk  # 候选答案分数
        l_order, r_order = [], []  # 组合顺序
        for i in range(topk):
            for j in range(topk):
                l_order.append(i)
                r_order.append(j)
        for i in range(maxlen_answer):  # 强制要求输出不超过maxlen字
            l_proba, r_proba = self.model.predict([session, xid, yl_id[l_order], yr_id[r_order], zid]) # 计算左右解码概率
            l_proba = l_proba[:, i, 3:]  # 直接忽略<padding>、<unk>、<start>
            r_proba = np.concatenate([r_proba[:, i, 2: 3], r_proba[:, i, 4:]], 1)  # 直接忽略<padding>、<unk>、<end>
            l_proba = l_proba.reshape((topk, topk, -1)).mean(1)  # 对所有候选R2L序列求平均，得到当前L2R方向的预测结果
            r_proba = r_proba.reshape((topk, topk, -1)).mean(0)  # 对所有候选L2R序列求平均，得到当前R2L方向的预测结果
            l_log_proba = np.log(l_proba + 1e-6)  # 取对数方便计算
            r_log_proba = np.log(r_proba + 1e-6)  # 取对数方便计算
            l_arg_topk = l_log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
            r_arg_topk = r_log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
            _yl_id, _yr_id = [], []  # 暂存的候选目标序列
            _l_scores, _r_scores = [], []  # 暂存的候选目标序列得分
            if i == 0:
                for j in range(topk):
                    _yl_id.append(list(yl_id[j]) + [l_arg_topk[0][j] + 3])
                    _l_scores.append(l_log_proba[0][l_arg_topk[0][j]])
                    _yr_id.append(list(yr_id[j]) + [r_arg_topk[0][j] + 3])
                    _r_scores.append(r_log_proba[0][r_arg_topk[0][j]])
            else:
                for j in range(topk):
                    for k in range(topk):  # 遍历topk*topk的组合
                        _yl_id.append(list(yl_id[j]) + [l_arg_topk[j][k] + 3])
                        _l_scores.append(l_scores[j] + l_log_proba[j][l_arg_topk[j][k]])
                        _yr_id.append(list(yr_id[j]) + [r_arg_topk[j][k] + 3])
                        _r_scores.append(r_scores[j] + r_log_proba[j][r_arg_topk[j][k]])
                _l_arg_topk = np.argsort(_l_scores)[-topk:]  # 从中选出新的topk
                _r_arg_topk = np.argsort(_r_scores)[-topk:]  # 从中选出新的topk
                _yl_id = [_yl_id[k] for k in _l_arg_topk]
                _l_scores = [_l_scores[k] for k in _l_arg_topk]
                _yr_id = [_yr_id[k] for k in _r_arg_topk]
                _r_scores = [_r_scores[k] for k in _r_arg_topk]
            yl_id = np.array(_yl_id)
            yr_id = np.array(_yr_id)
            l_scores = np.array(_l_scores)
            r_scores = np.array(_r_scores)
            l_ends = np.where(yl_id[:, -1] == 3)[0]
            r_ends = np.where(yr_id[:, -1] == 3)[0]
            if len(l_ends) > 0 and len(r_ends) == 0:
                k = l_scores[l_ends].argmax()
                return self.data_info.id2str(yl_id[l_ends[k]])
            if len(l_ends) == 0 and len(r_ends) > 0:
                k = r_scores[r_ends].argmax()
                return self.data_info.id2str(yr_id[r_ends[k]][::-1])
            if len(l_ends) > 0 and len(r_ends) > 0:
                lk = l_scores[l_ends].argmax()
                rk = r_scores[r_ends].argmax()
                if l_scores[l_ends][lk] > r_scores[r_ends][rk]:
                    return self.data_info.id2str(yl_id[l_ends[lk]])
                else:
                    return self.data_info.id2str(yr_id[r_ends[rk]][::-1])
        # 如果maxlen字都找不到<end>，直接返回
        lk = l_scores.argmax()
        rk = r_scores.argmax()
        if l_scores[lk] > r_scores[rk]:
            return self.data_info.id2str(yl_id[lk])
        else:
            return self.data_info.id2str(yr_id[rk][::-1])

    def to_one_hot(self, x, length = None):
        # 输出一个词表大小的向量，来标记该词是否在问题出现过
        if length == None:
            length = len(self.data_info.chars) + 4
        x, x_mask = x
        x = K.cast(x, 'int32')
        x = K.one_hot(x, length)
        x = K.sum(x_mask * x, 1, keepdims=True)
        x = K.cast(K.greater(x, 0.5), 'float32')
        return x
