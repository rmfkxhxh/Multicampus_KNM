#!/usr/bin/python
import tensorflow as tf
import sys
from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

tf.app.flags.DEFINE_string('f', '', 'kernel')

tf.app.flags.DEFINE_string('phase', 'train', 'The phase can be train, eval or test')

tf.app.flags.DEFINE_boolean('load', False, 'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.app.flags.DEFINE_boolean('train_cnn', True, 'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.app.flags.DEFINE_integer('beam_size', 3, 'The size of beam search for caption generation')

tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices.')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement') 

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', 'vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')


FLAGS = tf.app.flags.FLAGS
FLAGS(sys.argv, known_only=True)

def main(argv):
    with tf.device('/gpu:0'):
        config = Config()
        config.phase = FLAGS.phase
        config.train_cnn = FLAGS.train_cnn
        config.beam_size = FLAGS.beam_size
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        with tf.Session(config=session_conf) as sess:

            if FLAGS.phase == 'train':
                # training phase
                data = prepare_train_data(config)
                model = CaptionGenerator(config)
                sess.run(tf.global_variables_initializer())
                if FLAGS.load:
                    model.load(sess, FLAGS.model_file)
                if FLAGS.load_cnn:
                    model.load_cnn(sess, FLAGS.cnn_model_file)
                tf.Graph().finalize()
                model.train(sess, data)

            elif FLAGS.phase == 'eval':
                # evaluation phase
                coco, data, vocabulary = prepare_eval_data(config)
                model = CaptionGenerator(config)
                model.load(sess, FLAGS.model_file)
                tf.Graph().finalize()
                model.eval(sess, coco, data, vocabulary)

            else:
                # testing phase
                data, vocabulary = prepare_test_data(config)
                model = CaptionGenerator(config)
                model.load(sess, FLAGS.model_file)
                tf.Graph().finalize()
                model.test(sess, data, vocabulary)

if __name__ == '__main__':
    tf.app.run()
