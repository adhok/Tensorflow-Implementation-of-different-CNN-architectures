# Tensorflow-Implementation-of-different-CNN-architectures
Implementation using the book  "Hands on Machine Learning with Sci-Kit Learn Keras and Tensorflow" as a guide


**The following architectures are implemented**  


* GoogLeNet Architecture

* AlexNet architecture

* LeNet 5 architecture

* ResNet - 34 architecture

* ResNet - 50 architecture

* ResNet - 152 architecture

* VGGNet based on the paper https://arxiv.org/pdf/1409.1556.pdf ; In this repository I have implemented 16(C & D) layered and 19(E) layered version of the network.

* Xception Model 

* MobileNet

* MobileNetV2

* Squeeze and Excitation ResNet - 50 (SE ResNet - 50)

* SE - ResNeXt - 50


## How can you use these notebooks?

After downloading and running a notebook a `model` object is created. To run this model object on your own image data set you can run the following example.

This is assuming that the folder follows the structure below
```
train_folder
  - category_1_folder
  - category_2_folder
  - category_3_folder
test_folder
  - category_1_folder
  - category_2_folder
  - category_3_folder

```



Your images can be fed using the following snippet



```

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()
batch_size = 32
img_height = 32
img_width = 32


train_data = datagen.flow_from_directory('train_folder/', class_mode='sparse', target_size=(img_height, img_width), batch_size=batch_size, seed=123)

val_data = datagen.flow_from_directory('test_folder/', shuffle=False, class_mode='sparse', target_size=(img_height, img_width), batch_size=batch_size, seed=123)

import os

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.compile(loss = keras.losses.SparseCategoricalCrossentropy(),optimizer=keras.optimizers.SGD(learning_rate=0.0003),metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])



history = model.fit(train_data,epochs = 100,validation_data = val_data,callbacks=[cp_callback])



```



