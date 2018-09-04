import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput,DataInputTest,DataInputEval
from model import Model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class Args():
    is_training = False
    embedding_size = 64
    brand_list=None
    msort_list=None
    item_count=-1
    brand_count=-1
    msort_count=-1



if __name__ == '__main__':
    #read data
    with open('dataset.pkl', 'rb') as f:
        train_set_1 = pickle.load(f)
        train_set_2 = pickle.load(f)
        train_set_3 = pickle.load(f)
        test_set = pickle.load(f)
        brand_list = pickle.load(f)
        msort_list = pickle.load(f)
        user_count, item_count, brand_count, msort_count = pickle.load(f)
        item_key, brand_key, msort_key, user_key = pickle.load(f)
    print('user_count: %d\titem_count: %d\tbrand_count: %d\tmsort_count: %d' %
          (user_count, item_count, brand_count, msort_count))

    train_set = train_set_1 + train_set_2 + train_set_3
    print('train set size', len(train_set))

    #init args

    args=Args()
    args.brand_list=brand_list
    args.msort_list=msort_list
    args.item_count=item_count
    args.brand_count=brand_count
    args.msort_count=msort_count

    #else para
    epoch=3
    train_batch_size = 32
    test_batch_size = 50
    checkpoint_dir = 'save_path/ckpt'


    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        #build model
        model = Model(args)
        #init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sys.stdout.flush()
        if args.is_training:
            lr = 1.0
            start_time = time.time()
            for _ in range(epoch):

                random.shuffle(train_set)

                epoch_size = round(len(train_set) / train_batch_size)
                loss_sum = 0.0
                for _, uij in DataInput(train_set, train_batch_size):
                   
                    loss = model.train(sess, uij, lr)
                    loss_sum += loss

                    if model.global_step.eval() % 1000 == 0:
                        model.save(sess, checkpoint_dir)
                        print('Global_step %d\tTrain_loss: %.4f' %
                              (model.global_step.eval(),
                               loss_sum/1000))

                        print('Epoch %d Global_step %d\tTrain_loss: %.4f' %
                              (model.global_epoch_step.eval(), model.global_step.eval(),
                               loss_sum / 1000))
                        sys.stdout.flush()
                        loss_sum = 0.0
                    if model.global_step.eval() % 336000 == 0:
                        lr = 0.1

                print('Epoch %d DONE\tCost time: %.2f' %
                          (model.global_epoch_step.eval(), time.time() - start_time))
                sys.stdout.flush()
                model.global_epoch_step_op.eval()

        else:
            print('test')
            model.restore(sess, checkpoint_dir)
            '''
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state('./save_path')
            if ckpt and ckpt.model_checkpoint_path:
                print("Successfully loaded:", ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            '''
            out_file_skn = open("pred_skn.txt", "w")
    
            for _, uij in DataInputTest(test_set, test_batch_size):
                output = model.test(sess, uij[0], uij[1], uij[2],uij[3])
                pre_index = np.argsort(-output, axis=1)[:, 0:200]
             
                for y in range(len(uij[0])):
                    out_file_skn.write(str(uij[0][y]))
                    pre_skn = pre_index[y]
                    print(pre_skn)
                    for k in pre_skn:
                        out_file_skn.write("\t%i" % item_key[k])
                    out_file_skn.write("\n")


