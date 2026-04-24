import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ========================
# Sampling Layer
# ========================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ========================
# Encoder (Audio + Lyrics)
# ========================
def build_encoder(audio_shape, lyrics_dim, latent_dim):
    audio_in = keras.Input(shape=audio_shape, name="audio_input")
    lyrics_in = keras.Input(shape=(lyrics_dim,), name="lyrics_input")

    # Audio branch
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(audio_in)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)

    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    # Lyrics branch
    t = layers.Dense(128, activation="relu")(lyrics_in)
    t = layers.Dense(64, activation="relu")(t)

    # Fusion
    h = layers.Concatenate()([x, t])
    h = layers.Dense(256, activation="relu")(h)
    h = layers.Dropout(0.2)(h)

    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(
        [audio_in, lyrics_in],
        [z_mean, z_log_var, z],
        name="multimodal_beta_encoder"
    )

    return encoder, shape_before_flatten


# ========================
# Decoder
# ========================
def build_decoder(shape_before_flatten, latent_dim, lyrics_dim):
    z_in = keras.Input(shape=(latent_dim,), name="z_input")

    # Audio reconstruction
    x = layers.Dense(256, activation="relu")(z_in)
    x = layers.Dense(
        shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2],
        activation="relu"
    )(x)

    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

    audio_out = layers.Conv2D(
        1,
        3,
        padding="same",
        activation="sigmoid",
        name="audio_recon"
    )(x)

    # Lyrics reconstruction
    t = layers.Dense(128, activation="relu")(z_in)
    lyrics_out = layers.Dense(
        lyrics_dim,
        activation="linear",
        name="lyrics_recon"
    )(t)

    decoder = keras.Model(
        z_in,
        [audio_out, lyrics_out],
        name="multimodal_beta_decoder"
    )

    return decoder


# ========================
# Multimodal Beta-VAE
# ========================
class MultiModalBetaVAE(keras.Model):
    def __init__(self, encoder, decoder, beta=0.05, lyrics_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.lyrics_weight = lyrics_weight

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.audio_loss_tracker = keras.metrics.Mean(name="audio_recon_loss")
        self.lyrics_loss_tracker = keras.metrics.Mean(name="lyrics_recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.audio_loss_tracker,
            self.lyrics_loss_tracker,
            self.kl_loss_tracker,
        ]

    # ========================
    # Loss computation
    # ========================
    def compute_losses(self, audio, lyrics, training=False):
        z_mean, z_log_var, z = self.encoder([audio, lyrics], training=training)
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)

        audio_recon, lyrics_recon = self.decoder(z, training=training)
        audio_recon = audio_recon[:, :tf.shape(audio)[1], :tf.shape(audio)[2], :]

        audio_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(audio - audio_recon), axis=[1, 2, 3])
        )

        lyrics_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(lyrics - lyrics_recon), axis=1)
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )

        total_loss = audio_loss + self.lyrics_weight * lyrics_loss + self.beta * kl_loss

        return total_loss, audio_loss, lyrics_loss, kl_loss

    # ========================
    # FIXED TRAIN STEP
    # ========================
    def train_step(self, data):
        if isinstance(data, tuple):
            if len(data) == 1:
                data = data[0]

        audio = data[0]
        lyrics = data[1]

        audio = tf.cast(audio, tf.float32)
        lyrics = tf.cast(lyrics, tf.float32)

        audio = tf.where(tf.math.is_finite(audio), audio, tf.zeros_like(audio))
        lyrics = tf.where(tf.math.is_finite(lyrics), lyrics, tf.zeros_like(lyrics))

        with tf.GradientTape() as tape:
            total_loss, audio_loss, lyrics_loss, kl_loss = self.compute_losses(
                audio,
                lyrics,
                training=True
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.audio_loss_tracker.update_state(audio_loss)
        self.lyrics_loss_tracker.update_state(lyrics_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "audio_recon_loss": self.audio_loss_tracker.result(),
            "lyrics_recon_loss": self.lyrics_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    # ========================
    # FIXED TEST STEP
    # ========================
    def test_step(self, data):
        if isinstance(data, tuple):
            if len(data) == 1:
                data = data[0]

        audio = data[0]
        lyrics = data[1]

        audio = tf.cast(audio, tf.float32)
        lyrics = tf.cast(lyrics, tf.float32)

        audio = tf.where(tf.math.is_finite(audio), audio, tf.zeros_like(audio))
        lyrics = tf.where(tf.math.is_finite(lyrics), lyrics, tf.zeros_like(lyrics))

        total_loss, audio_loss, lyrics_loss, kl_loss = self.compute_losses(
            audio,
            lyrics,
            training=False
        )

        self.total_loss_tracker.update_state(total_loss)
        self.audio_loss_tracker.update_state(audio_loss)
        self.lyrics_loss_tracker.update_state(lyrics_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "audio_recon_loss": self.audio_loss_tracker.result(),
            "lyrics_recon_loss": self.lyrics_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }