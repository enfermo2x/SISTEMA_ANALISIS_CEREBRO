import tensorflow as tf
from src.utils.config import IMG_SIZE, CLASSES

def conv_block(x, filters, num_layers=2, kernel_size=3):
    for _ in range(num_layers):
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    return x

def build_cnn_custom(input_shape=(*IMG_SIZE, 1), num_classes=len(CLASSES)):
    inputs = tf.keras.Input(shape=input_shape)

    x = conv_block(inputs, 32, 2)
    x = conv_block(x, 64, 2)
    x = conv_block(x, 128, 2)
    x = conv_block(x, 256, 2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
