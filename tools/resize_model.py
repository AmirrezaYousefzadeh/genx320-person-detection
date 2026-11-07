"""Rebuild the fully convolutional st_yololcv1 Keras model for another input size (the weights do not depend on it;
the output grid becomes SIZE/16): resize_model.py IN.keras SIZE OUT.keras"""
import sys

import tensorflow as tf

src, size, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = tf.keras.models.load_model(src, compile=False)
cfg = m.get_config()
for layer in cfg["layers"]:
    if layer["class_name"] == "InputLayer":
        key = "batch_shape" if "batch_shape" in layer["config"] else "batch_input_shape"
        layer["config"][key] = [None, size, size, 3]
m2 = tf.keras.Model.from_config(cfg)
m2.set_weights(m.get_weights())
m2.save(dst)
print(f"{dst}: input {m2.input_shape}, output {m2.output_shape}")
