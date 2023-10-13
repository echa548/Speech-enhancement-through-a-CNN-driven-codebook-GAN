import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import os
import re
import random
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# Generator model
def build_generator():
    model = keras.Sequential()
    model.add(layers.Dense(1024, input_shape=(1024,)))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(8192, activation='relu'))
    model.add(layers.Dense(16384, activation='relu'))  # Adding an additional dense layer
    model.add(layers.Dense(1024 * 6))
    model.add(layers.Reshape((1024, 6)))
    return model


def build_discriminator():
    model = keras.Sequential()
    model.add(layers.Reshape((1024,), input_shape=(1024,)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    return model

# Define the WGAN-LP
class WGANLP(tf.keras.Model):
    def __init__(self, generator, discriminator, transform_fn):
        super(WGANLP, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.transform_fn = transform_fn

    def compile(self, g_optimizer, d_optimizer, lambda_penalty):
        super(WGANLP, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.lambda_penalty = lambda_penalty

    def call(self, inputs):
        return self.discriminator(inputs)

    def gradient_penalty(self, real_samples, generated_samples):
        batch_size = real_samples.shape[0]
        alpha = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=1.0)
        interpolated_samples = alpha * real_samples + (1 - alpha) * generated_samples

        with tf.GradientTape() as tape:
            tape.watch(interpolated_samples)
            predictions = self.discriminator(interpolated_samples)

        gradients = tape.gradient(predictions, interpolated_samples)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)

        return gradient_penalty

    def train_discriminator_step(self, real_samples, noise):
        with tf.GradientTape() as tape:
           
            generated_samples = self.generator(noise)

            # Transform the generated samples
            transformed_samples = self.transform_fn(generated_samples, noise)
            sum_real_samples = tf.reduce_sum(real_samples, axis=-1, keepdims=True)
            sum_transformed_samples = tf.reduce_sum(transformed_samples, axis=-1, keepdims=True)
            normalized_real_samples = real_samples / (sum_real_samples + 1e-20)  # Adding a small value to avoid division by zero
            normalized_transformed_samples = transformed_samples / (sum_transformed_samples + 1e-20)  # Adding a small value to avoid division by zero
            # row_index = 0
            # print("Sum of row", row_index, "for normalized_real_samples:", sum(normalized_real_samples[row_index].numpy()))

            # Train the discriminator
            d_predictions_real = self.discriminator(normalized_real_samples)
            d_predictions_fake = self.discriminator(normalized_transformed_samples)

            # Compute the Wasserstein distance (discriminator loss)
            d_loss = tf.reduce_mean(d_predictions_fake) - tf.reduce_mean(d_predictions_real)

            # Compute the gradient penalty
            gradient_penalty = self.gradient_penalty(normalized_real_samples, normalized_transformed_samples)

            # Add gradient penalty to discriminator loss
            d_loss += self.lambda_penalty * gradient_penalty

        # Compute gradients for discriminator
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        d_gradients = [(grad, var) for grad, var in zip(d_gradients, self.discriminator.trainable_variables) if grad is not None]

        # Apply gradients to discriminator optimizer
        self.d_optimizer.apply_gradients(zip([grad for grad, var in d_gradients], self.discriminator.trainable_variables))

        return {"d_loss": d_loss}

    def train_generator_step(self, noise):
        with tf.GradientTape() as tape:
            generated_samples = self.generator(noise)
            transformed_samples = self.transform_fn(generated_samples, noise)
            sum_transformed_samples = tf.reduce_sum(transformed_samples, axis=-1, keepdims=True)
            normalized_transformed_samples = transformed_samples / (sum_transformed_samples + 1e-20)  # Adding a small value to avoid division by zero
            g_predictions = self.discriminator(normalized_transformed_samples)
            g_loss = -tf.reduce_mean(g_predictions)

        # Compute gradients for generator
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        g_gradients = [(grad, var) for grad, var in zip(g_gradients, self.generator.trainable_variables) if grad is not None]

        # Apply gradients to generator optimizer
        self.g_optimizer.apply_gradients(zip([grad for grad, var in g_gradients], self.generator.trainable_variables))

        return {"g_loss": g_loss}

    def train_step(self, real_samples, noise):
        d_loss = self.train_discriminator_step(real_samples, noise)["d_loss"]
        g_loss = self.train_generator_step(noise)["g_loss"]

        return {"d_loss": d_loss, "g_loss": g_loss}

    def save(self, filepath, overwrite=True, save_format=None, options=None):
        # Save only the generator model
        self.generator.save(filepath, overwrite=overwrite, save_format=save_format, options=options)

# This transform is one of the ways that works, this is unintuitive though. This transform is what was used
# This was the one used for the report.
def transform_fn(samples, noisemix):

    Numpy_generated_sample = samples.numpy()
    Numpy_noisemix_sample = noisemix.numpy()
    Tensor_size = np.shape(Numpy_generated_sample)[0]
    Estimated_noise_PSD = np.zeros((Tensor_size, 1024))
    for Tensor in range(0, Tensor_size):
        Generated_sample = Numpy_generated_sample[Tensor, :, :]
        Generated_sample = Generated_sample.reshape((1024, 6))
        noise_sample = Numpy_noisemix_sample[Tensor, :]
        noise_sample = noise_sample.reshape(1024)
        Noise_inverse = np.linalg.pinv(np.abs(Generated_sample), rcond=1e-15)
        Noise_coeffs = np.transpose(Noise_inverse * noise_sample)
        Projection = (Noise_coeffs * np.abs(Generated_sample))
        Projection = np.asarray(Projection).clip(min=0)
        for Freq_bin in range(0, 1024):
            Estimated_noise_PSD[Tensor, Freq_bin] = np.sum(Projection[Freq_bin, :])
        Estimated_noise_PSD[Tensor, 512:1024] = np.flip(Estimated_noise_PSD[Tensor,0:512])

    transformed_samples = tf.cast(Estimated_noise_PSD, tf.float32)

    return transformed_samples

# Create generator and discriminator models
generator = build_generator()
discriminator = build_discriminator()

# Create WGAN-LP instance
wganlp = WGANLP(generator, discriminator, transform_fn)

# Compile the WGAN-LP
wganlp.compile(
    g_optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
    d_optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
    lambda_penalty=10
)


# Load the full npy files
real_samples = np.load('Clean_paired_speech_PSDs.npy')
noise = np.load('Speech_with_noise_PSDs.npy')



#set batch size if a larger mini-batch is desired.
batch_size = 64
num_batches = len(real_samples) // batch_size

save_interval = 1  # Interval for saving the model. Change this if needed.
#Increase iterations if discriminator needs to reach optimallity before training generator.
discriminator_iterations = 5  # Number of times to train the discriminator per epoch
os.chdir('..')
current_epoch = 0
#Change the names to the existing saved models. If need to train from scratch then just put whatever u want in the paths
#below.
generator_model_path = "Models/GAN-Models/Full_Curriculum_Speech_1000_generator"
discriminator_model_path = "Models/GAN-Models/Full_Curriculum_Speech_1000_discriminator"

if os.path.exists(generator_model_path) and os.path.exists(discriminator_model_path):
    saved_generator = tf.saved_model.load(generator_model_path)
    saved_discriminator = tf.saved_model.load(discriminator_model_path)
    
    if saved_generator and saved_discriminator:
        print("Loaded saved generator and discriminator models.")
        wganlp.generator = saved_generator
        wganlp.discriminator = saved_discriminator
        wganlp.compile(
            g_optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
            d_optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
            lambda_penalty=10)
        #wganlp.save(f"trained_wganlp_model_epoch_Dense_20")
        if os.path.exists("Models/GAN-Models/current_epoch_speech.txt"):
            with open("Models/GAN-Models/current_epoch_speech.txt", "r") as epoch_file:
                current_epoch = int(epoch_file.read())
                print(f"Resuming training from epoch {current_epoch}")
else:
    print("No saved generator and discriminator models found.")

os.chdir(dname)

for epoch in range(current_epoch,1500):
    print(f"Epoch {epoch + 1}/{1500}")

    for d_iter in range(discriminator_iterations):
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size

            real_batch = tf.convert_to_tensor(real_samples[start:end], dtype=tf.float32)
            noise_batch = tf.convert_to_tensor(noise[start:end], dtype=tf.float32)

            losses = wganlp.train_discriminator_step(real_batch, noise_batch)
            real_batch = 0
            noise_batch = 0
            print('Epoch:' + str(epoch+1) +' '+ 'Iteration:' +str(d_iter+1),  end =' ')
            print(losses)  # Print the losses for each batch

    # Train the generator
    for batch in range(num_batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        noise_batch = tf.convert_to_tensor(noise[start:end], dtype=tf.float32)

        losses = wganlp.train_generator_step(noise_batch)
        print('Epoch:' + str(epoch+1),  end =' ')
        print(losses)  # Print the losses for each batch
        

    if (epoch + 1) % save_interval == 0:
        os.chdir('..')
        model_filename = f"Full_Curriculum_Speech_{epoch + 1}"
        current_epoch = epoch + 1
        with open("Models/GAN-Models/current_epoch_speech.txt", "w") as epoch_file:
            epoch_file.write(str(current_epoch))
            
        # Save the generator and discriminator models
        tf.saved_model.save(wganlp.generator, f"Models/GAN-Models/{model_filename}_generator")
        tf.saved_model.save(wganlp.discriminator, f"Models/GAN-Models/{model_filename}_discriminator")
        print("Models saved.")
        os.chdir(dname)