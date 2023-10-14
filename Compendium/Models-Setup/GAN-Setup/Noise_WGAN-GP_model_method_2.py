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
    model.add(layers.Dense(1024 * 9))
    model.add(layers.Reshape((1024, 9)))
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

    def train_discriminator_step(self, real_samples, noise,observations):
        with tf.GradientTape() as tape:

            generated_samples = self.generator(noise)

            # Transform the generated samples
            transformed_samples = self.transform_fn(generated_samples, noise, observations)
            # Train the discriminator
            sum_real_samples = tf.reduce_sum(real_samples, axis=-1, keepdims=True)
            sum_transformed_samples = tf.reduce_sum(transformed_samples, axis=-1, keepdims=True)
            ratios = sum_real_samples / (sum_transformed_samples + 1e-20)  # Adding a small value to avoid division by zero
            scaled_transformed_samples = transformed_samples * ratios

            # Train the discriminator
            d_predictions_real = self.discriminator(real_samples)
            d_predictions_fake = self.discriminator(scaled_transformed_samples)

            # Compute the Wasserstein distance (discriminator loss)
            d_loss = tf.reduce_mean(d_predictions_fake) - tf.reduce_mean(d_predictions_real)

            # Compute the gradient penalty
            gradient_penalty = self.gradient_penalty(real_samples, scaled_transformed_samples)
            d_loss += self.lambda_penalty * gradient_penalty

        # Compute gradients for discriminator
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        d_gradients = [(grad, var) for grad, var in zip(d_gradients, self.discriminator.trainable_variables) if grad is not None]

        # Apply gradients to discriminator optimizer
        self.d_optimizer.apply_gradients(zip([grad for grad, var in d_gradients], self.discriminator.trainable_variables))

        return {"d_loss": d_loss}

    def train_generator_step(self, noise,observations,real_samples):
        with tf.GradientTape() as tape:
            generated_samples = self.generator(noise)
            transformed_samples = self.transform_fn(generated_samples, noise, observations)
            sum_real_samples = tf.reduce_sum(real_samples, axis=-1, keepdims=True)
            sum_transformed_samples = tf.reduce_sum(transformed_samples, axis=-1, keepdims=True)
            ratios = sum_real_samples / (sum_transformed_samples + 1e-60)  # Adding a small value to avoid division by zero
            scaled_transformed_samples = transformed_samples * ratios
            g_predictions = self.discriminator(scaled_transformed_samples)
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

# Define the arbitrary mathematical transform function
def transform_fn(samples, noisemix,observations):
    # Apply your arbitrary mathematical transform to generated samples here
    Numpy_generated_sample = samples.numpy()
    #print(np.shape(Numpy_generated_sample))
    Numpy_noisemix_sample = noisemix.numpy()
    mean_vector = observations.numpy()
    Tensor_size = np.shape(Numpy_generated_sample)[0]
    Estimated_noise_PSD = np.zeros((Tensor_size, 1024))
    for Tensor in range(0, Tensor_size):
        
        
        Generated_sample = Numpy_generated_sample[Tensor, :, :]
        Generated_sample = Generated_sample.reshape((1024, 9))
        Generated_sample = np.abs(Generated_sample)
        noise_sample = Numpy_noisemix_sample[Tensor, :]
        noise_sample = noise_sample.reshape(1024,1)
        Effect_on_mean = noise_sample*mean_vector
        Projection = Effect_on_mean*Generated_sample
        for Freq_bin in range(0, 1024):
            Estimated_noise_PSD[Tensor, Freq_bin] = np.sum(Projection[Freq_bin, :])
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


real_samples = np.load('Clean_PSD2.npy')
noise = np.load('Mixture_PSD2.npy')

text_file = open("Noise_list.txt", "r")  #make sure this is at the same location as this file
lines = text_file.readlines()
text_file.close()
Noise_codebook2 = np.zeros((1024,9))



for frequency_bin in range (0,len(lines)):
  string_list = lines[frequency_bin].split()
  for component in range (0, len(string_list)):
   Noise_codebook2[frequency_bin,component] = float(string_list[component])

test = tf.convert_to_tensor(Noise_codebook2, dtype=tf.float32) 

batch_size = 64
num_batches = len(real_samples) // batch_size
#num_batches = len(wow) // batch_size
save_interval = 1  # Interval for saving the model
discriminator_iterations = 5  # Number of times to train the discriminator per epoch

current_epoch = 0
generator_model_path = "SKALAR_True_distrib_1_generator"
discriminator_model_path = "SKALAR_True_distrib_1_discriminator"

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
        if os.path.exists("current_epoch.txt"):
            with open("current_epoch.txt", "r") as epoch_file:
                current_epoch = int(epoch_file.read())
                print(f"Resuming training from epoch {current_epoch}")
else:
    print("No saved generator and discriminator models found.")

for epoch in range(current_epoch,1500):
    print(f"Epoch {epoch + 1}/{1500}")

    for d_iter in range(discriminator_iterations):
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size

            real_batch = tf.convert_to_tensor(real_samples[start:end], dtype=tf.float32)
            noise_batch = tf.convert_to_tensor(noise[start:end], dtype=tf.float32)

            losses = wganlp.train_discriminator_step(real_batch, noise_batch,test)
            real_batch = 0
            noise_batch = 0
            print('Epoch:' + str(epoch+1) +' '+ 'Iteration:' +str(d_iter+1),  end =' ')
            print(losses)  # Print the losses for each batch

    # Train the generator
    for batch in range(num_batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        real_batch = tf.convert_to_tensor(real_samples[start:end], dtype=tf.float32)
        noise_batch = tf.convert_to_tensor(noise[start:end], dtype=tf.float32)

        losses = wganlp.train_generator_step(noise_batch,test,real_batch)
        print('Epoch:' + str(epoch+1),  end =' ')
        print(losses)  # Print the losses for each batch
        

    if (epoch + 1) % save_interval == 0:
        model_filename = f"SKALAR_True_distrib_{epoch + 1}"
        current_epoch = epoch + 1
        with open("current_epoch.txt", "w") as epoch_file:
            epoch_file.write(str(current_epoch))
            
        # Save the generator and discriminator models using tf.saved_model.save
        tf.saved_model.save(wganlp.generator, f"{model_filename}_generator")
        tf.saved_model.save(wganlp.discriminator, f"{model_filename}_discriminator")
        #wganlp.save(f"trained_wganlp_model_epoch_Dense_{epoch + 1}")
        print("Models saved.")