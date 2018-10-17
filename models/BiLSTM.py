import tensorflow as tf 
import tensorflow.contrib.slim as slim
import os
import time

# This model is from the paper --
# "Bowman S R, Angeli G, Potts C, et al. A large annotated corpus for learning natural language inference[J]. arXiv preprint arXiv:1508.05326, 2015."

class BiLSTM(object):
    def __init__(self, config):
        self.seq_len1 = config['premise_len'] # premise sequence length
        self.seq_len2 = config['hypothesis_len'] # premise sequence length
        self.embedding_size = config['embedding_size'] # embedding size
        self.vocabulary_size = config['vocabulary_size'] # vocabulary size
        self.l2_reg_lambda = config['l2_reg_lambda'] # l2 normalization weight
        self.hidden_size = 128  #lstm cell units
        self.n_layer = 1 #lstm layer number
        self.num_class = 1 # class number

        with tf.variable_scope('inputs1'):
            self.x1 = tf.placeholder(tf.int64, [None, self.seq_len1], name='premise_inputs')
            self.x2 = tf.placeholder(tf.int64, [None, self.seq_len2], name='hypothesis_inputs')
            self.y = tf.placeholder(tf.float32, [None, self.num_class], name='y_input')

        with tf.variable_scope('embedding'):
            self.w_embed = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name='w_embed')

        with tf.variable_scope('fc1'):
            self.W1 = tf.get_variable("W1",shape=[(self.hidden_size*2)*2, 1024],initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.Variable(tf.constant(0.1, shape=[1024]), name="b1")  
        
        with tf.variable_scope('fc2'):   
            self.W2 = tf.get_variable("W2",shape=[1024, 1024],initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.Variable(tf.constant(0.1, shape=[1024]), name="b2") 

        with tf.variable_scope('fc3'):   
            self.W3 = tf.get_variable("W3",shape=[1024, 1024],initializer=tf.contrib.layers.xavier_initializer())
            self.b3 = tf.Variable(tf.constant(0.1, shape=[1024]), name="b3")   

        with tf.variable_scope('outputs'):
            self.W = tf.get_variable("W",shape=[1024, 1],initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")

        self.scores = self.bilstm_inference()

        with tf.name_scope('loss'):
            self.probs = tf.nn.sigmoid(self.scores)
            self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.y)

            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)

            self.obj = tf.reduce_mean(self.losses) + self.l2_reg_lambda * l2_loss

    def lstm_cell(self):
        with tf.name_scope('lstm_cell'):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0)

    def bilstm(self, inputs, scope):
        with tf.variable_scope(scope):
            cells_fw = [self.lstm_cell() for _ in range(self.n_layer)]
            cells_bw = [self.lstm_cell() for _ in range(self.n_layer)]
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,dtype=tf.float32)
            outputs_c = tf.concat([inputs, outputs], axis = 2)
        return outputs_c

            
    def bilstm_inference(self):
        with tf.name_scope('bilstm_inference'):
            embed1 = tf.nn.embedding_lookup(self.ft_w_embed, self.x1) 
            embed2 = tf.nn.embedding_lookup(self.ft_w_embed, self.x2) 

            lstm_1 = self.bilstm(embed1,'bilstm_1')
            lstm_2 = self.bilstm(embed2,'bilstm_2')

            avg1 = tf.reduce_mean(lstm_1, axis = 1) 
            avg2 = tf.reduce_mean(lstm_2, axis = 1)
            
            avg = tf.concat([avg1, avg2], axis = 1)

            # dense layer 1
            fc1_mat = tf.nn.xw_plus_b(avg, self.W1, self.b1, name="fc1")
            fc1_act = tf.nn.tanh(fc1_mat)

            # dense layer 2
            fc2_mat = tf.nn.xw_plus_b(fc1_act, self.W2, self.b2, name="fc2")
            fc2_act = tf.nn.tanh(fc2_mat)

            # dense layer 3
            fc3_mat = tf.nn.xw_plus_b(fc1_act, self.W3, self.b3, name="fc3")
            fc3_act = tf.nn.tanh(fc3_mat)
            
            # classifer
            scores = tf.nn.xw_plus_b(fc3_act, self.W, self.b, name="classifer")
            return scores

    def model_summary():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)



if __name__ == '__main__':
    config = {
        'seq_len1':15,
        'seq_len2':15,
        'vocabulary_size':56755,
        'embedding_size':300,
        'l2_reg_lambda':0.0001,
        'num_epochs':100,
        'batch_size':300,
        'train_size':20000,
        'valid_size':5000,
    }

    # create model instance
    print('load model...')
    model = BiLSTM(config)
    model.model_summary()

    train_step = tf.train.AdamOptimizer(0.0005).minimize(model.obj)

    #init and train
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    rounds = 1
    with tf.Session() as sess:
        sess.run([init_global, init_local])
        while rounds <= config['num_epochs']:
            i = 1
            c_loss = 0.0
            while i <= int(config['train_size']/config['batch_size'])+1:
                # Load data
                print('load data batch...')
                train_q1, train_q2, train_y = read_trainset(i)
                i += 1
                c,_ = sess.run([model.obj, train_step], feed_dict={model.x1: train_q1, model.x2: train_q2, model.y: train_y})
                c_loss += c
            i = 0
            c_val_loss = 0.0
            probs = []
            label = []
            while i <= int(config['train_size']/config['batch_size'])+1:
                valid_q1, valid_q2, valid_y = read_valset(i)
                c_val_, probs_c = sess.run([model.obj, model.probs], feed_dict={model.x1: valid_q1, model.x2: valid_q2, model.y: valid_y})
                c_val_loss += c_val_
                probs = np.concatenate((probs, probs_c[:,0]))
                label = np.concatenate((label, valid_y[:,0]))

            auc_s_sklearn = roc_auc_score(label, probs)
            t = time.strftime(format, time.localtime())
            print t, "---round num %5d  : train_loss:%.5f   val_loss:%.5f  val_auc:%.5f"%(rounds, c_loss, c_val_loss, auc_s_sklearn)
            rounds += 1
