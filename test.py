import tensorflow as tf
import matplotlib.pyplot as plt

# print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

# On charge les images pour entrainer le réseau et tester le réseau
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Pour voir la première image des images de test
plt.imshow(test_images[0])
plt.show()

# Les pixels des images sont su 256 bits, il faut les normaliser pour que ça marche bien (je ne sais pas pourquoi,
# sûrement de la TNI)
training_images = training_images / 255.0  # normalizing
test_images = test_images / 255.0  # normalizing

# Permet de créer notre réseau de neuronnes, ici un réseau séquentiel (1 variable d'entrée et 1 variable de sortie si
# j'ai bien compris)
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Pour entrainer notre réseau
model.fit(training_images, training_labels, epochs=5)

# Pour tester le réseau
model.evaluate(test_images, test_labels)

# classification = model.predict(test_images)

# print(classification[0])
# print(mnist)
