import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data (0-1 range)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Define the label types
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the size of the encoded (latent) representation
encoding_dim = 64  # Dimension of the latent space

# Define the Autoencoder model
# Flatten the 28x28 images into vectors of size 784
input_img = Input(shape=(28, 28))
flattened = Flatten()(input_img) # (784, )

# Encoder part
encoded = Dense(128, activation='relu')(flattened)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)  # Output layer of the encoder

# Decoder part
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)  # Output layer of the decoder
decoded = Reshape((28, 28))(decoded)  # Reshape the decoded vector back to 28x28

# Combine encoder and decoder into the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Define the encoder model to extract latent representations
encoder = Model(input_img, encoded)

# Get the latent representation of the test images
latent_representations = encoder.predict(x_test)

# Use t-SNE to further reduce dimensionality to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(latent_representations)

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
colors = np.random.rand(len(tsne_results[:, 0]))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=2, c=y_test, cmap="tab10")
plt.title('t-SNE Visualization of Latent Space (Fashion-MNIST)')
plt.savefig('t-SNE.png')
plt.show()

# Latent feature extraction using the encoder
latent_train = encoder.predict(x_train)
latent_test = encoder.predict(x_test)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build a simple classifier on top of the latent space
input_latent = Input(shape=(64,))
classifier_output = Dense(64, activation='relu')(input_latent)
classifier_output = Dense(32, activation='relu')(classifier_output)
classifier_output = Dense(10, activation='softmax')(classifier_output)

classifier = Model(input_latent, classifier_output)

# Compile the classifier model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classifier and capture training history
history = classifier.fit(latent_train, y_train, 
                         epochs=50, 
                         batch_size=256, 
                         shuffle=True, 
                         validation_data=(latent_test, y_test))

# Plotting the loss and accuracy curves for training and validation
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training.png')
plt.show()

# Evaluate the classifier on the test set
test_loss, test_acc = classifier.evaluate(latent_test, y_test, verbose=0)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Predict on the test set
y_pred = np.argmax(classifier.predict(latent_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Display some correct and incorrect predictions
correct_indices = np.nonzero(y_pred == y_true)[0]
incorrect_indices = np.nonzero(y_pred != y_true)[0]

# Plot correct predictions
plt.figure(figsize=(12, 6))
for i, correct in enumerate(correct_indices[:5]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[correct].reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[y_true[correct]]}\nPred: {class_names[y_pred[correct]]}")
    plt.axis('off')

# Plot incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:5]):
    plt.subplot(2, 5, i+6)
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[y_true[incorrect]]}\nPred: {class_names[y_pred[incorrect]]}")
    plt.axis('off')

plt.savefig('predictions.png')
plt.show()