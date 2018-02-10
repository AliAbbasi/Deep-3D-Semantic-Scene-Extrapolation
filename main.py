import tensorflow as tf

#  ---------------------------------------------------------------------------------------------------------------------


flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Training Epochs")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam, default: 0.5")
flags.DEFINE_integer("batch_size", 100, "The training batch size, default: 100")
flags.DEFINE_string("input_pattern", "*.npy", "Input data extension for glob, default: *.npy")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints, default: checkpoint")
flags.DEFINE_string("output_dir", "output", "Directory name to save the results [output]")
flags.DEFINE_boolean("train", True, "True for training, False for testing, default: False")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing, default: False")
flags.DEFINE_integer("generate_test_scene", 100, "Number of scenes to generate during test, default: 100")
flags.DEFINE_string("model", "SEGAN", "The model type include: SEGAN and SECNN, default: SEGAN")
FLAGS = flags.FLAGS

#  ---------------------------------------------------------------------------------------------------------------------


def main(_):
    print(flags.FLAGS.__flags)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        if FLAGS.model == "SEGAN":
            pass
            # create SEGAN
        else:
            pass
            # create SECNN

        if FLAGS.train:
            pass
        else:
            pass
            # load the model parameters, then test

#  ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    tf.app.run()

    