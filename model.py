    #dataset creation
    from keras.preprocessing.image import ImageDataGenerator
    training_directory = 'Dataset/Train data'
    test_directory = 'Dataset/Test data'
    import numpy as np
    import cv2
    import os

    trainingdata_generator = ImageDataGenerator(horizontal_flip=True,
                                                vertical_flip=True,
                                                zca_whitening=True,
                                                zoom_range=0.2,
                                                rotation_range=20,
                                                rescale=1./255,
                                                shear_range=0.2,
                                                fill_mode='nearest')

    validationdata_generator = ImageDataGenerator(rescale=1./255,
                                                  horizontal_flip=True,
                                                  vertical_flip=True,
                                                  zca_whitening=True,
                                                  zoom_range=0.2,
                                                  rotation_range=20)



    training_set = trainingdata_generator.flow_from_directory(training_directory,
                                                              target_size=(112,112),
                                                              class_mode='categorical',
                                                              batch_size=16,
                                                              shuffle=True)

    validation_set = validationdata_generator.flow_from_directory(test_directory,
                                                                  target_size=(112,112),
                                                                  class_mode='categorical',
                                                                  batch_size=16,
                                                                  shuffle=True)
    #cnn model

    import keras
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.layers import Activation
    from keras.layers import Dropout
    from keras.layers import BatchNormalization

    bird_classifier = Sequential()
    bird_classifier.add(Conv2D(filters=48,kernel_size=(3,3),input_shape=(112,112,3),strides=1,kernel_initializer='glorot_uniform',data_format="channels_last",
    padding="same"))
    bird_classifier.add(BatchNormalization())
    bird_classifier.add(Activation("relu"))
    bird_classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
    bird_classifier.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,kernel_initializer='glorot_uniform',padding="same"))
    bird_classifier.add(BatchNormalization())
    bird_classifier.add(Activation("relu"))
    bird_classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
    bird_classifier.add(Conv2D(filters=80,kernel_size=(3,3),strides=1,kernel_initializer='glorot_uniform',padding="same"))
    bird_classifier.add(BatchNormalization())
    bird_classifier.add(Activation("relu"))
    bird_classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
    bird_classifier.add(Conv2D(filters=96,kernel_size=(3,3),strides=1,kernel_initializer='glorot_uniform',padding="same"))
    bird_classifier.add(BatchNormalization())
    bird_classifier.add(Activation("relu"))
    bird_classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
    bird_classifier.add(Conv2D(filters=112,kernel_size=(3,3),strides=1,kernel_initializer='glorot_uniform',padding="same"))
    bird_classifier.add(BatchNormalization())
    bird_classifier.add(Activation("relu"))
    bird_classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
    bird_classifier.add(Flatten())
    bird_classifier.add(Dense(2048,activation="relu",kernel_initializer="glorot_uniform"))
    bird_classifier.add(Dropout(0.5))
    bird_classifier.add(Dense(2048,activation="relu",kernel_initializer="glorot_uniform"))
    bird_classifier.add(Dropout(0.5))
    bird_classifier.add(Dense(10,activation="softmax",kernel_initializer="glorot_uniform"))

    bird_classifier.summary()
    bird_classifier.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

    bird_classifier.fit_generator(training_set,
                                  steps_per_epoch = 30,
                                  epochs = 50,
                                  validation_data = validation_set,
                                  validation_steps=8)

    bird_classifier.save('bird_image_classifier.h5')

    def test_sample():
        image_dir="D:\major_project"
        path=os.path.join(image_dir,'rw.jpg')
        img_array=cv2.imread(path)
        new_array = cv2.resize(img_array,(112,112))
        new_array = new_array/255.0
        new_array = np.array(new_array).reshape(1,112,112,3)
        return new_array

    test_image = test_sample()

    probabilities = bird_classifier.predict_proba(test_image)
    for i in range(0,10):
        print(probabilities[0][i])
    print(bird_classifier.predict_classes(test_image))
    for i in training_set.class_indices:
        print(i)
    steps = 100
    predictions = model.predict_generator(validation_generator, steps=steps)
