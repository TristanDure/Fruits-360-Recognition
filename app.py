"""
Fruits-360 水果识别 — Flask Web 应用
上传图片 → 自动去背景 → 模型推理 → 返回 Top-5 预测
"""
import os
import io
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
from rembg import remove
from model import get_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== 加载模型 =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), 'class_names.txt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 260

print(f'设备: {DEVICE}')
print(f'加载模型: {MODEL_PATH}')

model = get_model(num_classes=NUM_CLASSES, freeze_backbone=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print('模型加载完成！')

# 加载类别名
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = [line.strip().split(': ', 1)[1] for line in f if line.strip()]

# ===== 图片预处理（和训练时测试集一致） =====
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def remove_background(pil_img):
    """
    用 rembg (U2Net) 去掉图片背景，贴到白底上
    这样处理后的图片和 Fruits-360 训练集风格一致
    """
    # 去掉背景 → RGBA 透明图
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    output = remove(img_bytes)  # 返回 PNG bytes

    # 加载为 RGBA 图
    rgba = Image.open(io.BytesIO(output)).convert('RGBA')

    # 贴到白色背景上（模拟 Fruits-360 白底）
    white_bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    white_bg.paste(rgba, mask=rgba)

    return white_bg.convert('RGB')


def predict_image(image_path, top_k=5):
    """对一张图片进行预测，返回 top_k 结果"""
    pil_img = Image.open(image_path).convert('RGB')

    # ① 去背景 → 白底（和训练集一致）
    processed = remove_background(pil_img)

    # ② 保存处理后的图片（方便调试查看效果）
    name, ext = os.path.splitext(os.path.basename(image_path))
    processed_path = os.path.join(UPLOAD_FOLDER, f'{name}_nobg.jpg')
    processed.save(processed_path, quality=90)

    # ③ 推理
    tensor = transform(processed).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for i in range(top_k):
        name = class_names[top_indices[i].item()]
        prob = round(top_probs[i].item() * 100, 2)
        results.append({'name': name, 'prob': prob})

    return results, processed_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'})

    # 保存上传的图片
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        results, processed_path = predict_image(filepath, top_k=5)
        return jsonify({
            'top1_name': results[0]['name'],
            'top1_prob': results[0]['prob'],
            'top5': results
        })
    except Exception as e:
        return jsonify({'error': f'识别失败: {str(e)}'})


if __name__ == '__main__':
    print('=' * 50)
    print('🍎 水果智能识别系统')
    print(f'   260种水果蔬菜 | ResNet-18 迁移学习 | 自动去背景')
    print(f'   访问地址: http://localhost:5000')
    print('=' * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
