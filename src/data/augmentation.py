import tensorflow as tf
from src.utils.config import IMG_SIZE, BATCH_SIZE, SEED

def augment_image(x, y):
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_flip_up_down(x, seed=SEED)
    x = tf.image.random_brightness(x, max_delta=0.1, seed=SEED)
    x = tf.image.random_contrast(x, lower=0.9, upper=1.1, seed=SEED)
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, seed=SEED)
    x = x + noise
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y

def create_train_dataset(X_train, y_train, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train), seed=SEED)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_val_dataset(X_val, y_val, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_test_dataset(X_test, y_test, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_tf_generators(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=BATCH_SIZE):
    train_ds = create_train_dataset(X_train, y_train, batch_size)
    val_ds = create_val_dataset(X_val, y_val, batch_size)
    test_ds = create_test_dataset(X_test, y_test, batch_size)
    return train_ds, val_ds, test_ds
