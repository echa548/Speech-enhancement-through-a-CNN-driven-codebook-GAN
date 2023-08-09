import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
import os

os.chdir('C:/Users/Timothy/Documents/GitHub/COPY_OF_CURRENT_VER/Spectogram')

# Generator model
def build_generator():
    model = keras.Sequential()
    model.add(layers.Dense(1024, input_shape=(1024,)))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(8192, activation='relu'))
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

    def train_discriminator_step(self, real_samples, noise):
        with tf.GradientTape() as tape:
            # Generate fake sample
            generated_samples = self.generator(noise)

            # Transform the generated samples
            transformed_samples = self.transform_fn(generated_samples, noise)

            # Train the discriminator
            d_predictions_real = self.discriminator(real_samples)
            d_predictions_fake = self.discriminator(transformed_samples)

            # Compute the Wasserstein distance (discriminator loss)
            d_loss = tf.reduce_mean(d_predictions_fake) - tf.reduce_mean(d_predictions_real)

            # Compute the gradient penalty
            gradient_penalty = self.gradient_penalty(real_samples, transformed_samples)

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
            g_predictions = self.discriminator(transformed_samples)
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
def transform_fn(samples, noisemix):
    # Apply your arbitrary mathematical transform to generated samples here
    Numpy_generated_sample = samples.numpy()
    #print(np.shape(Numpy_generated_sample))
    Numpy_noisemix_sample = noisemix.numpy()
    Tensor_size = np.shape(Numpy_generated_sample)[0]
    Estimated_noise_PSD = np.zeros((Tensor_size, 1024))
    for Tensor in range(0, Tensor_size):
        Generated_sample = Numpy_generated_sample[Tensor, :, :]
        Generated_sample = Generated_sample.reshape((1024, 9))
        noise_sample = Numpy_noisemix_sample[Tensor, :]
        noise_sample = noise_sample.reshape(1024)
        Noise_inverse = np.linalg.pinv(Generated_sample, rcond=1e-15)
        Noise_coeffs = np.transpose(Noise_inverse * noise_sample)
        #print(np.shape(Noise_coeffs))
       # Projection = np.zeros((1024,9))
        Projection = (Noise_coeffs * Generated_sample)
        Projection = np.asarray(Projection).clip(min=0)
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
    lambda_penalty=1.0
)

# Generate real samples with shape (data_point, 1024)
real_samples = tf.convert_to_tensor(np.load('Clean_PSD.npy'), dtype=tf.float32)
# Generate user-defined noise with shape (data_point, 1024)
noise = tf.convert_to_tensor(np.load('Mixture_PSD.npy'), dtype=tf.float32)

# Train the WGAN-LP
batch_size = 64
num_batches = len(real_samples) // batch_size
save_interval = 100  # Interval for saving the model
discriminator_iterations = 5  # Number of times to train the discriminator per epoch
saved_models = [file for file in os.listdir('.') if file.startswith('trained_wganlp_model_epoch_Dense_500')]
if saved_models:
    saved_models.sort()
    latest_model = saved_models[-1]
    start_epoch = int(latest_model.split('_')[-1]) + 1
    wganlp.generator = tf.keras.models.load_model(latest_model)

    # Compile the generator and discriminator (critic)
    wganlp.generator.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
        
    )

    wganlp.discriminator.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.00005),
        
    )

for epoch in range(1500):
    print(f"Epoch {epoch + 1}/{1500}")

    for _ in range(discriminator_iterations):
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size

            real_batch = real_samples[start:end]
            noise_batch = noise[start:end]

            losses = wganlp.train_discriminator_step(real_batch, noise_batch)
            print(losses)  # Print the losses for each batch

    # Train the generator
    for batch in range(num_batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size

        noise_batch = noise[start:end]

        losses = wganlp.train_generator_step(noise_batch)
        print(losses)  # Print the losses for each batch

    if (epoch + 1) % save_interval == 0:
        wganlp.save(f"trained_wganlp_model_epoch_Dense_{epoch + 1}")

wganlp.save("trained_wganlp_model")
