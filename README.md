# Handwritten Digit Recognition System

A Class 12 CBSE Artificial Intelligence project. Draw a digit from 0 to 9 and a trained Artificial Neural Network predicts it.

## Features
- Interactive drawing canvas
- Image preprocessing (grayscale and 8 × 8 resizing)
- Neural-network classification
- Prediction confidence and probability chart
- Model test accuracy displayed in the app

## Technology
- Python
- Streamlit
- NumPy and Pillow
- Scikit-learn (`load_digits`, `MLPClassifier`)

## Run on your computer

1. Install Python 3.10 or newer.
2. Extract this project folder.
3. Open Command Prompt/Terminal inside the folder.
4. Create a virtual environment (recommended):

   Windows:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

6. Start the app:
   ```bash
   streamlit run app.py
   ```

The app will open in your browser. Draw a digit and click **Predict digit**.

## Dataset note
This version uses scikit-learn's built-in handwritten digits dataset. Each source image is 8 × 8 pixels and represents one of ten classes (0–9). This keeps the project lightweight and suitable for a school demonstration. It is not the full MNIST dataset.

## Files
- `app.py` — complete application and model training code
- `requirements.txt` — required Python packages
- `PROJECT_REPORT.md` — school project report content
- `VIVA_QUESTIONS.md` — sample viva questions and answers
