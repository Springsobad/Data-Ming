import numpy as np
import pickle
from flask import Flask, render_template, request

# Tải mô hình đã huấn luyện
with open('decision_tree.pkl', 'rb') as f:
    model = pickle.load(f)

# Tải encoding mapping đã được huấn luyện trước
encoding_mapping = {
    'odor': {'a': 0, 'c': 1, 'f': 2, 'l': 3, 'm': 4, 'n': 5, 'p': 6, 's': 7, 'y': 8},
    'spore_print_color': {'b': 0, 'h': 1, 'k': 2, 'n': 3, 'o': 4, 'r': 5, 'u': 6, 'w': 7, 'y': 8},
    'gill_color': {'b': 0, 'e': 1, 'g': 2, 'h': 3, 'k': 4, 'n': 5, 'o': 6, 'p': 7, 'r': 8, 'u': 9, 'w': 10, 'y': 11},
    'stalk_surface_above_ring': {'f': 0, 'k': 1, 's': 2, 'y': 3},
    'stalk_surface_below_ring': {'f': 0, 'k': 1, 's': 2, 'y': 3},
    'ring_type': {'e': 0, 'f': 1, 'l': 2, 'n': 3, 'p': 4},
    'class': {'e': 0, 'p': 1}
}
label_mapping = {
    'odor': {
        'a': 'Hạnh nhân', 'l': 'Hồi', 'c': 'Creosote', 'y': 'Cá', 'f': 'Thối',
        'm': 'Mốc', 'n': 'Không có', 'p': 'Hắc', 's': 'Cay'
    },
    'spore_print_color': {
        'k': 'Đen', 'n': 'Nâu', 'b': 'Vàng nhạt', 'h': 'Sô cô la',
        'r': 'Xanh lá', 'o': 'Cam', 'u': 'Tím', 'w': 'Trắng', 'y': 'Vàng'
    },
    'gill_color': {
        'k': 'Đen', 'n': 'Nâu', 'b': 'Vàng nhạt', 'h': 'Sô cô la',
        'g': 'Xám', 'r': 'Xanh lá', 'o': 'Cam', 'p': 'Hồng',
        'u': 'Tím', 'e': 'Đỏ', 'w': 'Trắng', 'y': 'Vàng'
    },
    'stalk_surface_above_ring': {
        'f': 'Sợi', 'y': 'Vảy', 'k': 'Mượt', 's': 'Mịn'
    },
    'stalk_surface_below_ring': {
        'f': 'Sợi', 'y': 'Vảy', 'k': 'Mượt', 's': 'Mịn'
    },
    'ring_type': {
        'e': 'Tạm thời', 'f': 'Mở rộng', 'l': 'Lớn', 'n': 'Không có', 'p': 'Móc'
    }
}


# Khởi tạo ứng dụng Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận các giá trị từ form
    odor = request.form['odor']
    spore_print_color = request.form['spore_print_color']
    gill_color = request.form['gill_color']
    stalk_surface_above_ring = request.form['stalk_surface_above_ring']
    stalk_surface_below_ring = request.form['stalk_surface_below_ring']
    ring_type = request.form['ring_type']

    # Mã hóa các giá trị chuỗi thành số sử dụng encoding_mapping
    encoded_data = []
    for feature, value in zip(['odor', 'spore_print_color', 'gill_color', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'ring_type'],
                              [odor, spore_print_color, gill_color, stalk_surface_above_ring, stalk_surface_below_ring, ring_type]):
        # Kiểm tra nếu giá trị tồn tại trong encoding_mapping
        if value in encoding_mapping[feature]:
            encoded_value = encoding_mapping[feature].get(value)
            encoded_data.append(encoded_value)
        else:
            # Xử lý nếu giá trị không có trong encoding_mapping (có thể trả về giá trị mặc định hoặc báo lỗi)
            return f"Error: The value '{value}' for {feature} is not valid."

    # Tạo input mẫu cho mô hình
    sample = np.array([encoded_data])

    # Dự đoán với mô hình
    prediction = model.predict(sample)

    # Kết quả dự đoán
    if prediction[0] == 0:  # Poisonous
        result = "Poisonous Mushroom"
        image = "nam_doc.jpg"
    else:  # Edible
        result = "Edible Mushroom"
        image = "nam_an_duoc.jpg"

    return render_template('index.html', result=result, image=image,
                           odor=odor,
                           spore_print_color=spore_print_color,
                           gill_color=gill_color,
                           stalk_surface_above_ring=stalk_surface_above_ring,
                           stalk_surface_below_ring=stalk_surface_below_ring,
                           ring_type=ring_type,
                           odor_label=label_mapping['odor'][odor],
                           spore_print_color_label=label_mapping['spore_print_color'][spore_print_color],
                           gill_color_label=label_mapping['gill_color'][gill_color],
                           stalk_surface_above_ring_label=label_mapping['stalk_surface_above_ring'][
                               stalk_surface_above_ring],
                           stalk_surface_below_ring_label=label_mapping['stalk_surface_below_ring'][
                               stalk_surface_below_ring],
                           ring_type_label=label_mapping['ring_type'][ring_type])


if __name__ == "__main__":
    app.run(debug=True)
