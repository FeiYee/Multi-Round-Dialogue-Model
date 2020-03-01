from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

from keras_layer_normalization import LayerNormalization

from WordsHandel.Word2Vec import BuildWord2Vec
from SelfLayers.Attention import Attention
from SelfLayers.Bidirectional import Bidirectional
from SelfLayers.ModulatedLayerNormalization import ModulatedLayerNormalization
from SelfLayers.ScaleShift import ScaleShift


class BuildModel:
    def __init__(self, chars_num=None, w2v_file=None):
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
            self.word2vec = BuildWord2Vec(w2v_file)
        self.model = self.build()

    def to_one_hot(self, x, length=None):
        # 输出一个词表大小的向量，来标记该词是否在问题出现过
        if length == None:
            length = len(self.data_info.chars) + 4
        x, x_mask = x
        x = K.cast(x, 'int32')
        x = K.one_hot(x, length)
        x = K.sum(x_mask * x, 1, keepdims=True)
        x = K.cast(K.greater(x, 0.5), 'float32')
        return x

    def seq_maxpool(self, x):
        """
        seq格式：[None, seq_len, s_size]
        mask格式：[None, seq_len, 1]
        先除去mask部分，然后再做maxpooling。
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
        x, yl, yr, z = x_in, yl_in, yr_in, z_in

        session_level = Lambda(lambda x: x[:, 0])(session_info)
        session_years = Lambda(lambda x: x[:, 1])(session_info)
        session_kinds = Lambda(lambda x: x[:, 2])(session_info)
        session_station = Lambda(lambda x: x[:, 3])(session_info)

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
        x = Bidirectional(CuDNNLSTM(self.z_dim // 2, return_sequences=True))([x, x_mask])
        x = LayerNormalization()(x)
        x = Bidirectional(CuDNNLSTM(self.z_dim // 2, return_sequences=True))([x, x_mask])

        z = LayerNormalization()(z)
        z = Bidirectional(CuDNNLSTM(self.z_dim // 2, return_sequences=True))([z, z_mask])
        z = LayerNormalization()(z)
        z = Bidirectional(CuDNNLSTM(self.z_dim // 2, return_sequences=True))([z, z_mask])

        x_max = Lambda(self.seq_maxpool)([x, x_mask])

        session = LayerNormalization()(session)
        session = CuDNNLSTM(self.z_dim // 4, return_sequences=True)(session)
        session = LayerNormalization()(session)
        session = CuDNNLSTM(self.z_dim // 4, return_sequences=True)(session)
        session = LayerNormalization()(session)
        session = CuDNNLSTM(self.z_dim // 4, return_sequences=True)(session)

        # 正向decoder，单向LSTM
        y = embedding(yl)
        y = ModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = CuDNNLSTM(self.z_dim, return_sequences=True)(y)
        y = ModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = CuDNNLSTM(self.z_dim, return_sequences=True)(y)
        yl = ModulatedLayerNormalization(self.z_dim // 4)([y, x_max])

        # 逆向decoder，单向LSTM
        y = embedding(yr)
        y = ModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = CuDNNLSTM(self.z_dim, return_sequences=True)(y)
        y = ModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = CuDNNLSTM(self.z_dim, return_sequences=True)(y)
        yr = ModulatedLayerNormalization(self.z_dim // 4)([y, x_max])

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
        model.compile(optimizer=Adam(self.learning_rate))
        # print(model.summary())
        return model

    def gen_sent(self, s, session, h='', topk=3,
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
        xid = np.array([self.data_info.str2id(s, maxlen_query)] * topk ** 2)  # 输入转id
        yl_id = np.array([[2]] * topk)  # L2R解码均以<start>开头，这里<start>的id为2
        yr_id = np.array([[3]] * topk)  # R2L解码均以<end>开头，这里<end>的id为3
        zid = np.array([self.data_info.str2id(h[::-1], maxlen_history)] * topk ** 2)  # 历史转id
        session = np.array([session] * topk ** 2)
        l_scores, r_scores = [0] * topk, [0] * topk  # 候选答案分数
        l_order, r_order = [], []  # 组合顺序
        for i in range(topk):
            for j in range(topk):
                l_order.append(i)
                r_order.append(j)
        for i in range(maxlen_answer):  # 强制要求输出不超过maxlen字
            l_proba, r_proba = self.model.predict([session, xid, yl_id[l_order], yr_id[r_order], zid])  # 计算左右解码概率
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