from tensorflow import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 加载模型
model = load_model('hch_model.h5')

def predict_image(image_path):
    img = Image.open(image_path).convert('L')  # 转为灰度图
    img = img.resize((28, 28))  # 调整为28x28大小
    img_arr = np.array(img).reshape(1, 784) / 255.0  # 转为数组并归一化
    prediction = model.predict(img_arr)
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label[0]

pred = predict_image("./test/0.png")
print(f"Predicted label: {pred}")

