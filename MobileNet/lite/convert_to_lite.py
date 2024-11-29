import tensorflow as tf
model = tf.keras.models.load_model('E:\\Programming\\Farmers\\MobileNet\\model_20240815-153554.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('E:\\Programming\\Farmers\\MobileNet\\lite\\model.tflite', 'wb') as f:
    f.write(tflite_model)