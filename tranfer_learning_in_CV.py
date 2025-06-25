import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def create_transfer_model(num_classes):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model





def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    return train_datagen, val_datagen


def train_transfer_model(train_dir, val_dir, num_classes, epochs=10, batch_size=32):
    model = create_transfer_model(num_classes)

    train_datagen, val_datagen = create_data_generators()
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs
    )
    return model, history