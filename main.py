# Useful libraries
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# List all files in the 'Images' directory
# import os
# for dirname, _, filenames in os.walk('Images'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Loading data
train_data = pd.read_csv("Images/sign_mnist_train/sign_mnist_train.csv")
test_data = pd.read_csv("Images/sign_mnist_test/sign_mnist_test.csv")
new_test_data = pd.read_csv("Images/new_data/new_data.csv")
# Pour l'analyse
test = pd.read_csv("Images/sign_mnist_test/sign_mnist_test.csv")
y = test['label']

# Permet de voir les 5 premières valeurs des valeurs de train_data
# print(train_data.head(10))

# Permet de voir la valeur du 3ᵉ pixel des valeurs de train_data
# print(train_data['pixel3'])

# Permet de voir le nombre d'images correspondant aux différents signes (désignés par un label de 0 à 24) dans le set
# dans train_data et test_data

sns.countplot(train_data['label'])
plt.title('Count of images of each sign in the training dataset')
plt.show()
sns.countplot(test_data['label'])
plt.title('Count of images of each sign in the testing dataset')
plt.show()

# On crée un vecteur y des labels des signes
y_train = train_data['label']
y_test = test_data['label']
y_new_test = new_test_data['label']
del train_data['label']
del test_data['label']
del new_test_data['label']

# À l'air de binariser les labels, mais je ne sais pas à quoi ça sert
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)
y_new_test = label_binarizer.fit_transform(y_new_test)

# On crée un vecteur x des valeurs des pixels des signes
x_train = train_data.values
x_test = test_data.values
x_new_test = new_test_data.values

# On normalise les data
x_train = x_train / 255
x_test = x_test / 255
x_new_test = x_new_test / 255

# On passe les data de 1-D à 3-D comme demandé dans un réseau à convolution(alias CNN)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_new_test = x_new_test.reshape(-1, 28, 28, 1)

# Preview des 10 premières images de train_data
f, ax = plt.subplots(2, 5)
f.set_size_inches(10, 10)
k = 0  # Si vous voulez voir d'autres images, c'est sur cet index qu'il faut jouer
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(x_train[k].reshape(28, 28), cmap="gray")
        k += 1
    plt.tight_layout()
plt.show()

# Pour ne pas faire de sur apprentissage on altère les images à entrainer en appliquant quelques
# transformations à celles-ci

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

# Le réseau de neurone à convolution tmtc

# Entrainement du model
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=24, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

nbr_epochs = 20
history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=nbr_epochs, validation_data=(x_test, y_test),
                    callbacks=[learning_rate_reduction])

print("La précision du modèle est de ", model.evaluate(x_test, y_test)[1] * 100, "%")
print("La précision du modèle avec les nouveaux tests est de ", model.evaluate(x_new_test, y_new_test)[1] * 100, "%")

# Analyses après l'entrainement du modèle

epochs = [i for i in range(nbr_epochs)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16, 9)

ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
ax[1].plot(epochs, val_loss, 'r-o', label='Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

predict_x = model.predict(x_test)
predictions = np.argmax(predict_x, axis=1)
print(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 9:
        predictions[i] += 1
predictions[:5]
classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y, predictions, target_names=classes))

cm = confusion_matrix(y, predictions)
cm = pd.DataFrame(cm, index=[i for i in range(25) if i != 9], columns=[i for i in range(25) if i != 9])
plt.figure(figsize=(15, 15))
sns.heatmap(cm, cmap="Greens", linecolor='black', linewidth=1, annot=True, fmt='')
plt.show()
