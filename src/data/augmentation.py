import tensorflow as tf
from src.utils.config import IMG_SIZE, BATCH_SIZE, SEED

def rotate_image_tf(image, angle):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cx, cy = w / 2.0, h / 2.0
    cos_a = tf.cos(-angle)
    sin_a = tf.sin(-angle)
    a, b, c = cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a
    d, e, f = sin_a, cos_a, cy * (1 - cos_a) - cx * sin_a
    transforms = [[a, b, c, d, e, f, 0.0, 0.0]]
    x = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=transforms,
        output_shape=tf.shape(image)[:2],
        fill_value=0.0,
        interpolation='BILINEAR'
    )
    return tf.squeeze(x, 0)

def augment_image(x, y):
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_flip_up_down(x, seed=SEED)
    angles = tf.random.uniform([], -0.35, 0.35, seed=SEED)
    x = rotate_image_tf(x, angles)
    x = tf.image.random_brightness(x, max_delta=0.15, seed=SEED)
    x = tf.image.random_contrast(x, lower=0.8, upper=1.2, seed=SEED)
    scale = tf.random.uniform([], 0.85, 1.15, seed=SEED)
    new_h = tf.cast(tf.cast(IMG_SIZE[0], tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(IMG_SIZE[1], tf.float32) * scale, tf.int32)
    x = tf.image.resize(x, (new_h, new_w))
    x = tf.image.resize_with_crop_or_pad(x, IMG_SIZE[0], IMG_SIZE[1])
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.015, seed=SEED)
    x = x + noise
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y

def create_train_dataset(X_train, y_train, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train), seed=SEED)
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
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
