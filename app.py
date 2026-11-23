import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="wide")

@st.cache_resource
def train_model():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=700,
        random_state=42,
        early_stopping=True
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, accuracy = train_model()

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit from **0 to 9** in the black box. The machine-learning model will predict the digit.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Draw your digit")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
    )
    predict_button = st.button("Predict digit", type="primary", use_container_width=True)

with right:
    st.subheader("AI prediction")
    if predict_button:
        if canvas_result.image_data is None or np.max(canvas_result.image_data[:, :, :3]) < 10:
            st.warning("Please draw a digit before clicking Predict.")
        else:
            # The training images contain bright digits on a dark background.
            rgba = canvas_result.image_data.astype(np.uint8)
            image = Image.fromarray(rgba, mode="RGBA").convert("L")
            image = image.resize((8, 8), Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32)

            # Remove faint canvas noise and scale pixel values to the dataset range (0–16).
            pixels[pixels < 35] = 0
            sample = (pixels / 255.0 * 16.0).reshape(1, -1)

            probabilities = model.predict_proba(sample)[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])

            st.metric("Predicted digit", str(prediction))
            st.progress(confidence)
            st.write(f"Model confidence: **{confidence * 100:.1f}%**")
            st.caption("Confidence is the model's estimated probability, not a guarantee that the prediction is correct.")
            st.subheader("Prediction probabilities")
            st.bar_chart({str(i): float(probabilities[i]) for i in range(10)})
            st.image(image.resize((240, 240)), caption="Processed 8 × 8 input", clamp=True)

st.divider()
st.subheader("About this project")
a, b, c = st.columns(3)
a.metric("Classes", "10", "Digits 0–9")
b.metric("Input size", "8 × 8", "64 pixel values")
c.metric("Test accuracy", f"{accuracy * 100:.1f}%")
st.markdown("""
**How it works**
1. The user draws a digit on the canvas.
2. The application converts the drawing into a grayscale image.
3. The image is resized to 8 × 8 pixels.
4. Pixel values are given to a trained Artificial Neural Network.
5. The model predicts the most likely digit and displays the result.

**Important:** This is an educational demonstration trained on scikit-learn's built-in handwritten-digits dataset. It may make mistakes, especially with unusual handwriting or drawings that differ from the training examples.
""")
st.caption("Class 12 CBSE Artificial Intelligence Project | Python + Machine Learning + Streamlit")
