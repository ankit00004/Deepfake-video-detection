from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import torch
from model.resnext_lstm import ResNeXtLSTM
from torchvision import transforms
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNeXtLSTM()
model.load_state_dict(torch.load("model/resnext_lstm.pth", map_location=device))
model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Helper: extract frames
def extract_frames(video_path, seq_len=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // seq_len)

    for i in range(seq_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frames.append(transform(frame))
    cap.release()

    # Pad if too few frames
    while len(frames) < seq_len:
        frames.append(frames[-1])

    return torch.stack(frames[:seq_len])  # [T, C, H, W]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return 'No video uploaded', 400

    video = request.files['video']
    if video.filename == '':
        return 'No selected video', 400

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    # Run prediction
    with torch.no_grad():
        frames = extract_frames(video_path).unsqueeze(0).to(device)  # shape: [1, T, C, H, W]
        output = model(frames)
        _, predicted = torch.max(output.data, 1)
        prediction = 'Real' if predicted.item() == 0 else 'Fake'

    return render_template('result.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
