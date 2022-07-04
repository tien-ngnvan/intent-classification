import tensorflow as tf
import official.nlp.optimization

from official import nlp


def softmax_loss(y_true, y_pred):
    # y_true: sparse target
    # y_pred: logist
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)
    return tf.reduce_mean(ce)


def optimizer(learning_rate, epochs, batch_size, len_data, warm_up=0.06):

    train_data_size = len_data
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = num_train_steps * warm_up

    decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=num_train_steps,
        end_learning_rate=0)

    warmup_schedule = nlp.optimization.WarmUp(
        initial_learning_rate=learning_rate,
        decay_schedule_fn=decay_schedule,
        warmup_steps=warmup_steps)

    optimizer = nlp.optimization.AdamWeightDecay(
        learning_rate=warmup_schedule,
        weight_decay_rate=0.01,
        epsilon=1e-8,
        beta_2=0.98,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    # overshoot.
    return optimizer