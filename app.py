import streamlit as st
import cv2
from PIL import Image
import numpy as np
from deepface import DeepFace

# 絵文字画像の読み込み
emoji_path = "emoji.png"
try:
    emoji = Image.open(emoji_path)
    st.write("Emoji loaded successfully.")
except Exception as e:
    st.error(f"Error loading emoji: {e}")

# 画像の読み込み
def load_image(image_file):
    img = Image.open(image_file)
    # 画像が8ビットのグレースケールまたはRGB形式かどうかを確認し、必要に応じて変換
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

# 画像内の顔の位置検出
def detect_faces(image):
    image_np = np.array(image)
    results = DeepFace.detectFace(image_np, detector_backend='opencv', enforce_detection=False)
    face_locations = []
    for result in results:
        x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
        face_locations.append((y, x+w, y+h, x))
    return face_locations

# 検出した顔に絵文字を重ねる
def apply_emoji(image, face_locations, emoji):
    image_np = np.array(image)
    for (top, right, bottom, left) in face_locations:
        emoji_resized = emoji.resize((right - left, bottom - top))
        emoji_np = np.array(emoji_resized)
        for c in range(0, 3):
            image_np[top:bottom, left:right, c] = emoji_np[:, :, c] * (emoji_np[:, :, 3] / 255.0) + image_np[top:bottom, left:right, c] * (1.0 - emoji_np[:, :, 3] / 255.0)
    return Image.fromarray(image_np)

# UI
st.title("顔認識スタンプアプリ")
st.write("画像をアップロードするか、カメラで写真を撮影してください。")
st.write("人物の顔を自動で認識して絵文字スタンプで隠します。")

# 画像のアップロード
image_file = st.file_uploader("画像を選択", type=["jpg", "jpeg", "png"])

# カメラで写真を撮影
if st.button("カメラを起動"):
    image_file = st.camera_input("カメラ")

if image_file is not None:
    image = load_image(image_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    face_locations = detect_faces(image)
    st.write(f"検出された顔の数: {len(face_locations)}")

    if face_locations:
        result_image = apply_emoji(image, face_locations, emoji)
        st.image(result_image, caption="スタンプを適用した画像", use_column_width=True)
    else:
        st.write("顔が検出されませんでした。")

# Streamlitアプリケーションの実行部分
if __name__ == "__main__":
    st.set_page_config(page_title="顔認識スタンプアプリ")
    st.write("Streamlitアプリケーションを実行しています")
