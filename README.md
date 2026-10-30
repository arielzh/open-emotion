# open-emotion

Facial-expression classification CNN using TensorFlow and Keras. Trained on AffectNet and FER2013 data. Live webcam demo.

ABSTRACT

Hotels are a critical site for human trafficking operations, and we lack front-line staff
trained to identify signs of abuse. Victims may display fleeting micro-expressions of distress, yet
they surpass human perception and go unnoticed.

However, machines have the ability to detect these split-second subtleties.
Using baseline single-network CNN, I built an AI model to identify facial expressions. I
used two datasets: FER2013 with smaller grayscale images, and AffectNet larger colored images.
Because of its small size and manageability, I used FER2013 to test different neural
network models. I evaluated the models based on validation loss, which calculated the distance
between the algorithm’s current output versus the expected output. The closer the loss is to zero,
the closer the algorithm models the dataset.

Algorithms that eliminated certain connections between neurons (DropOut) and
down-sampled input data (MaxPooling2D) scored lower losses than algorithms that preserved all
connections. A 4 convolutional-layer model with DropOut and MaxPooling in each layer was
most successful. Like a sieve, this approach of regularization extracted the most relevant
information; the structure improved the efficiency of subsequent processing and prevented
overfitting, where the model memorizes data instead of abstract reasoning.

After training on both datasets, the model reached a validation accuracy of 70%,
exceeding human-level identification accuracy of 65%.

Further experiments will integrate facial landmarks to increase model accuracy and add
emotional variables such as valence and arousal to create a nuanced spectrum of emotions.
