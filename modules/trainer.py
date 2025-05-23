##########################
# trainer.py
##########################

import tensorflow as tf
import tensorflow_transform as tft
from tfx_bsl.public import tfxio

LABEL_KEY = 'label'

def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=32):
    """Membaca TFRecords hasil transformasi dan memisahkan fitur serta label."""
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=LABEL_KEY
        ),
        schema=tf_transform_output.transformed_metadata.schema
    )
    return dataset

def _build_model(vocab_size=10000, embedding_dim=64, max_len=50):
    """Membangun model menggunakan Functional API agar kompatibel dengan serving."""
    inputs = tf.keras.Input(shape=(max_len,), name='token_ids')
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x) # Output biner (0 atau 1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # Loss untuk biner
        metrics=['accuracy']
    )
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membuat fungsi serving yang menerima serialized tf.Example sebagai input."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_tf_examples):
        # Ambil feature spec dan hapus label
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        # Parse serialized example
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        # Transform dengan TFT
        transformed_features = model.tft_layer(parsed_features)

        # Prediksi dengan model
        # Pastikan output sesuai dengan signature yang diharapkan (misalnya, probabilities)
        # Jika model outputnya (None, 1), kita bisa langsung return atau reshape jika perlu
        # Untuk TF Serving, seringkali output dictionary lebih disukai
        predictions = model(transformed_features['token_ids'])
        return {'output_0': predictions} # Nama output bisa disesuaikan

    return serve_tf_examples_fn

def run_fn(fn_args):
    """Fungsi utama yang dipanggil oleh komponen Trainer TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Bangun model
    model = _build_model()

    # Dataset
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=32
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=32
    )

    # Callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3
        )
    ]

    # Training
    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks
    )

    # Simpan model dengan serving dan eval signature
    serving_fn = _get_serve_tf_examples_fn(model, tf_transform_output)
    
    # Dapatkan signature evaluasi dari serving function juga (umumnya sama untuk klasifikasi)
    eval_fn = serving_fn

    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures={
            'serving_default': serving_fn,
            'eval': eval_fn
        }
    )
