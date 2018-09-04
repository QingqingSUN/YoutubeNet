import tensorflow as tf

class Model(object):

  def __init__(self,args):

    # self.sess=sess
    self.is_training=args.is_training
    # self.input_size=args.input_size
    self.embedding_size=args.embedding_size
    # self.basic_size=args.basic_size
    self.brand_list=args.brand_list
    self.msort_list=args.msort_list
    self.item_count=args.item_count
    self.brand_count=args.brand_count
    self.msort_count=args.msort_count
    self.build_model()



  def build_model(self):
    #placeholder
    self.u = tf.placeholder(tf.int32, [None,]) # user idx [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None]) # history click[B, T]
    self.sl = tf.placeholder(tf.int32, [None,]) # history len [B]
    self.last = tf.placeholder(tf.int32, [None, ])  # last click[B]
    self.basic=tf.placeholder(tf.float32,[None,None])#user basic feature[B,basic_size]
    self.sub_sample = tf.placeholder(tf.int32, [None, None])  # soft layer (pos_clict,neg_list)[B,sub_size]
    self.y = tf.placeholder(tf.float32, [None, None])  # label one hot[B]
    self.lr = tf.placeholder(tf.float64, [])


    #emb variable
    item_emb_w = tf.get_variable("item_emb_w", [self.item_count, self.embedding_size])
    item_b = tf.get_variable("item_b", [self.item_count],initializer=tf.constant_initializer(0.0))
    brand_emb_w = tf.get_variable("brand_emb_w", [self.brand_count, self.embedding_size])
    msort_emb_w = tf.get_variable("msort_emb_w", [self.msort_count, self.embedding_size])


    brand_list=tf.convert_to_tensor(self.brand_list,dtype=tf.int32)
    msort_list=tf.convert_to_tensor(self.msort_list,dtype=tf.int32)

    #historty seq
    hist_b=tf.gather(brand_list,self.hist_i)
    hist_m=tf.gather(msort_list,self.hist_i)

    h_emb=tf.concat([tf.nn.embedding_lookup(item_emb_w,self.hist_i),
                        tf.nn.embedding_lookup(brand_emb_w,hist_b),
                        tf.nn.embedding_lookup(msort_emb_w,hist_m)],axis=2)
    #historty mask
    mask=tf.sequence_mask(self.sl,tf.shape(h_emb)[1],dtype=tf.float32)#[B,T]
    mask=tf.expand_dims(mask,-1)#[B,T,1]
    mask=tf.tile(mask,[1,1,tf.shape(h_emb)[2]])#[B,T,3*e]

    h_emb*=mask#[B,T,3*e]
    hist=tf.reduce_sum(h_emb,1)#[B,3*e]
    hist=tf.div(hist,tf.cast(tf.tile(tf.expand_dims(self.sl,1),[1,3*self.embedding_size]),tf.float32))#[B,3*e]
    #last
    last_b=tf.gather(brand_list,self.last)
    last_m=tf.gather(msort_list,self.last)
    l_emb=tf.concat([tf.nn.embedding_lookup(item_emb_w,self.last),
                     tf.nn.embedding_lookup(brand_emb_w,last_b),
                     tf.nn.embedding_lookup(msort_emb_w,last_m)],axis=1)
    #net input
    self.input=tf.concat([hist,l_emb],axis=-1)
    # print('',)

    # dd net
    bn=tf.layers.batch_normalization(inputs=self.input,name='b1')
    layer_1=tf.layers.dense(bn,1024,activation=tf.nn.relu,name='f1')
    layer_2=tf.layers.dense(layer_1,512,activation=tf.nn.relu,name='f2')
    layer_3=tf.layers.dense(layer_2,3*self.embedding_size,activation=tf.nn.relu,name='f3')

    #softmax
    if self.is_training:
        sa_b=tf.gather(brand_list,self.sub_sample)
        sa_m=tf.gather(msort_list,self.sub_sample)

        sample_w=tf.concat([tf.nn.embedding_lookup(item_emb_w,self.sub_sample),
                             tf.nn.embedding_lookup(brand_emb_w,sa_b),
                             tf.nn.embedding_lookup(msort_emb_w,sa_m)],axis=2)#[B,sample,3*e]
        #sample_w=tf.nn.embedding_lookup(item_emb_w,self.sub_sample)
        sample_b=tf.nn.embedding_lookup(item_b,self.sub_sample)#[B,sample]
        user_v=tf.expand_dims(layer_3,1)#[B,1,3*e]
        sample_w=tf.transpose(sample_w,perm=[0,2,1])#[B,3*e,sample]
        self.logits=tf.squeeze(tf.matmul(user_v,sample_w),axis=1)+sample_b

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        '''
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
           )
        '''
        self.yhat = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(-self.y * tf.log(self.yhat + 1e-24))

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)


    else:
        all_emb=tf.concat([item_emb_w,
                           tf.nn.embedding_lookup(brand_emb_w,brand_list),
                           tf.nn.embedding_lookup(msort_emb_w,msort_list)],axis=1)
        self.logits=tf.matmul(layer_3,all_emb,transpose_b=True)+item_b
        self.output=tf.nn.softmax(self.logits)

  def train(self, sess, uij,l):
    loss,_= sess.run([self.loss,self.train_op], feed_dict={
        self.u: uij[0],
        self.sub_sample: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.last:uij[5],
        self.lr: l,
        })
    return loss
  def test(self, sess, uid, hist_i, sl,last):
    return sess.run(self.output, feed_dict={
        self.u: uid,
        self.hist_i: hist_i,
        self.sl: sl,
        self.last:last
        })
  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

