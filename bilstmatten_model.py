import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


class TBiLSTMConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 100  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    vocab_size = 5000  # 词汇表达小

    num_layers = 1  # 隐藏层层数
    hidden_dim = [256, 128]  # 隐藏层神经元
    rnn = 'lstm'  # lstm 或 gru

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    use_pre_trained = True
    emb_file = './data/wiki_100.utf8'
    tensorboard_dir = 'tensorboard/text_bilstm_atten'

class TBiLSTM(object):
    """文本分类，RNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.initializer = initializers.xavier_initializer()
        self.rnn()


    def _attention(self, _outputs):
        H = _outputs[0] + _outputs[1]
        output_reshape = tf.reshape(H, [-1, self.config.hidden_dim[-1]])
        w = tf.Variable(tf.random_normal([self.config.hidden_dim[-1]], stddev=0.1))
        w_reshape = tf.reshape(w, [-1, 1])
        M = tf.matmul(output_reshape, w_reshape)
        M_shape = tf.reshape(M, [-1, self.config.seq_length])
        self.alpha = tf.nn.softmax(M_shape)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.seq_length, 1]))
        sequeezeR = tf.squeeze(r)
        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        # output = tf.nn.dropout(sentenceRepren, self.keep_prob)

        return tf.reshape(sentenceRepren, [-1, self.config.hidden_dim[-1]])

    def rnn(self):
        """rnn模型"""

        def lstm_cell(hidden_dim):  # lstm核
            return tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)

        def gru_cell(hidden_dim):  # gru核
            return tf.contrib.rnn.GRUCell(hidden_dim)

        def dropout(hidden):  # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell(hidden)
            else:
                cell = gru_cell(hidden)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络

            for idx, hidden in enumerate(self.config.hidden_dim):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    lstm_fw_cell = dropout(hidden)
                    lstm_bw_cell = dropout(hidden)
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                             embedding_inputs, dtype=tf.float32,
                                                                             scope="bi-lstm" + str(idx))
                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2] 作为下一层的输入
                    embedding_inputs = tf.concat(outputs, 2)

            # 取出最后时间步的输出作为全连接的输入
            # last = embedding_inputs[:, -1, :]

            last = self._attention(outputs)

        # add attention:

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim[-1], name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
