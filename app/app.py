from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    # Đọc thông tin của ảnh
    img = Image.open(file)
    img2 = Image.open(file)
    #img = img.convert("RGB")
    img1 = img.convert("L")
    
    img = img1.resize((180, 180))  # Thay đổi kích thước ở đây
    # Chuyển ảnh đã thay đổi kích thước thành mảng numpy
    img = np.array(img)
    img = img/255

    img = img.reshape(1,180,180,1)
  
    # load model
    model = tf.keras.models.load_model(r"cnn.h5")
    
    label = ["NORMAL", "PNEUMONIA"]
    y_pred = model.predict(img)
    index=0
    if y_pred > 0.8:
        index = 1
    else:
        index = 0
    result = label[index]
    
    img_byte_array = io.BytesIO()
    img2.save(img_byte_array, format='JPEG')
    img_bytes = img_byte_array.getvalue()

    # Mã hóa dữ liệu bytes thành Base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    print(img_base64)
    #result = 0
    return render_template('index.html', img=img_base64, y_pred=y_pred, result=result)
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
