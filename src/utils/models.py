import tensorflow as tf


def get_vgg_16_model(image_size, base_model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=image_size,
        weights='imagenet',
        include_top=False
    )
    model.save(base_model_path)
    return model


def prepare_full_model(base_model, learning_rate, CLASSES=2, freeze_all=True,
                       freeze_till=None) -> tf.keras.models.Model:
    if freeze_all:
        for layer in base_model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in base_model.layers[:-freeze_till]:
            layer.trainable = False
    flatten_in = tf.keras.layers.Flatten()(base_model.output)
    prediction = tf.keras.layers.Dense(
        units=CLASSES,
        activation="softmax"
    )(flatten_in)
    full_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=prediction
    )
    full_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print(full_model.summary())
    return full_model
