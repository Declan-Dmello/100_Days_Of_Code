import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


class TransferLearningVGG16:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )






    def feature_extraction_approach(self):
        self.base_model.trainable = False

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=self.base_model.input, outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model










    def partial_fine_tuning_approach(self):

        self.base_model.trainable = False

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=self.base_model.input, outputs=output)

        for layer in self.base_model.layers[-4:]:
            layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=5e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model








    def full_fine_tuning_approach(self):

        self.base_model.trainable = True

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=self.base_model.input, outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


