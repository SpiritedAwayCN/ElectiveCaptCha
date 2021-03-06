from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import ResNetV2

def main():
    model = ResNetV2()
    model.build(input_shape=(None, 52, 52, 1))
    print(model.summary())

    train_datagen = ImageDataGenerator(
        samplewise_center=True, 
        samplewise_std_normalization=True, 
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1)

    test_datagen = ImageDataGenerator(
        samplewise_center=True, 
        samplewise_std_normalization=True)

    train_generator = train_datagen.flow_from_directory(
        'train',
        color_mode ='grayscale',
        target_size=(52, 52),
        batch_size = 32)

    test_generator = test_datagen.flow_from_directory(
        'test',
        color_mode ='grayscale',
        target_size=(52, 52),
        batch_size = 32)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        epochs=15,
        validation_data=test_generator)
    
    model.save_weights('resnet_model2.h5')


if __name__ == "__main__":
    main()