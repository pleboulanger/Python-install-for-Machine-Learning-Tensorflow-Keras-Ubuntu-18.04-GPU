# Python-install-for-Machine-Learning-Tensorflow-Keras-Ubuntu-18.04-GPU
Python install for Machine Learning Tensorflow Keras Ubuntu 18.04 GPU

###### What you will need :

- Ubuntu 18.04
- Sudo rights
- A Cuda compatible Nvidia graphic card (GPU)

Hello! This post will aim at installing Python for Machine Learning with Keras on Ubuntu 18.04.
At the end of this tutorial you will have installed the following:
- Docker
- nvidia-docker v2
- Tensorflow Docker image including keras

First, we need to install Docker.
Go to https://docs.docker.com/install/ and download the Docker CE (Community Edition) for Ubuntu. (Direct link to Ubuntu install: https://docs.docker.com/install/linux/docker-ce/ubuntu/#os-requirements)

Install the program.

Docker is now installed on your computer. You have to add Docker in the root user group in order to use it without "sudo" at the beginning of every command line. (This gives Docker the sudo rights on your PC by the way)

In order to do that, enter the following commands in your terminal
>sudo groupadd docker

>sudo usermod -aG docker $USER

Reboot your computer.


And try it with
>docker run hello-world

If it downloads something without an error massage, it's running okay.
Now, you need to install nvidia-docker in order to use your GPU. Go to the following link and follow the install for Ubuntu 18.04

https://github.com/NVIDIA/nvidia-docker

Nvidia-docker is now installed!

Open a terminal
Run
>docker run --runtime=nvidia --rm -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3

You will have some lines saying that you launched a Jupyter server and giving a token like this:

http://127.0.0.1:8888/?token=1cf746371223c18dff95a99b4dfb7db57e14a1528e5c78e

For me, the token is: 1cf746371223c18dff95a99b4dfb7db57e14a1528e5c78e

Enter the following address in a web browser
>localhost:8888

You will be asked to enter your token.

Once it is done, you're good to go inside your Jupyter Notebook!


You can try to run the following "hello world"
```
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

The result should be something like that showing the training:

![alt text](https://github.com/pleboulanger/Python-install-for-Machine-Learning-Tensorflow-Keras-Ubuntu-18.04/blob/master/MNIST.PNG)

###### Sources :
https://hub.docker.com/r/tensorflow/tensorflow/

https://docs.docker.com/install/linux/linux-postinstall/

https://github.com/NVIDIA/nvidia-docker

https://www.tensorflow.org/install/docker
