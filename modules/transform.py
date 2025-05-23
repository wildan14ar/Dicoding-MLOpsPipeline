import tensorflow as tf
import tensorflow_transform as tft

# Nama kolom
TEXT_FEATURE_KEY = 'message'
LABEL_KEY = 'label'
MAX_SEQ_LEN = 50

def preprocessing_fn(inputs):
    # Ambil teks dan label
    text = inputs[TEXT_FEATURE_KEY]
    label = inputs[LABEL_KEY]

    # Preprocessing teks: lowercase dan hapus karakter khusus
    text_clean = tf.strings.regex_replace(tf.strings.lower(text), r'[^\w\s]', '')
    text_clean = tf.squeeze(text_clean, axis=-1)

    # Tokenisasi
    tokens = tf.strings.split(text_clean)

    # Konversi token menjadi ID (vocabulary dengan top_k=10000)
    token_ids = tft.compute_and_apply_vocabulary(tokens, top_k=10000)

    # Ubah token ke tensor padat
    token_ids_padded = token_ids.to_tensor(default_value=0)

    # Potong/pad ke panjang tetap
    token_ids_fixed_len = token_ids_padded[:, :MAX_SEQ_LEN]
    pad_len = tf.maximum(0, MAX_SEQ_LEN - tf.shape(token_ids_fixed_len)[1])
    token_ids_fixed_len = tf.pad(token_ids_fixed_len, [[0, 0], [0, pad_len]])

    # Pastikan shape eksplisit
    token_ids_fixed_len = tf.ensure_shape(token_ids_fixed_len, [None, MAX_SEQ_LEN])

    # Konversi label ke int64
    label = tf.cast(label, tf.int64)

    return {
        'token_ids': token_ids_fixed_len,
        LABEL_KEY: label
    }
