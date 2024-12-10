import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Load the trained model
model = load_model("captcha_solver_model.keras")

char_set = 'abcdefghijklmnopqrstuvwxyz0123456789'  # Character set
img_shape = (50, 200)  # Input shape for the model (height, width)

def preprocess_image(image_path):
    """
    Preprocess the selected image for the model:
    - Converts to grayscale
    - Resizes to model's input shape
    - Normalizes pixel values
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, img_shape)
        image = image / 255.0  # Normalize to [0, 1]
        return image.reshape(1, *img_shape, 1)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def predict_captcha(image_path):
    """
    Predict the CAPTCHA text using the trained model.
    """
    try:
        image = preprocess_image(image_path)
        predictions = model.predict(image)
        captcha_text = ''.join([char_set[np.argmax(p)] for p in predictions])
        return captcha_text
    except Exception as e:
        raise ValueError(f"Error predicting CAPTCHA: {e}")

def upload_file():
    """
    Handle file upload and display the prediction.
    """
    try:
        file_path = filedialog.askopenfilename()  # Open file dialog
        if file_path:
            # Extract true label if present in the filename (e.g., "abcd.png" => true label: "abcd")
            true_label = file_path.split("/")[-1].split(".")[0]

            # Predict CAPTCHA
            prediction = predict_captcha(file_path)

            # Display the uploaded image
            img = Image.open(file_path)
            img = img.resize((300, 75))  # Resize for display in Tkinter
            img = ImageTk.PhotoImage(img)
            img_label.configure(image=img)
            img_label.image = img
            img_label.pack(padx=10, pady=10)  # Show the label dynamically

            # Display the true label and prediction
            result_label.config(
                text=f"True: {true_label}\nPrediction: {prediction}",
                font=("Helvetica", 16),
                fg="green",
            )
    except Exception as e:
        # Display error message in the result label
        result_label.config(text=f"Error: {str(e)}", font=("Helvetica", 16), fg="red")


# Initialize Tkinter window
app = tk.Tk()
app.title("CAPTCHA Solver")

# Configure window size and style
app.geometry("600x400")
app.configure(bg="#f0f0f0")  # Light gray background

# Add a title label
title_label = tk.Label(
    app,
    text="CAPTCHA Solver",
    font=("Helvetica", 24, "bold"),
    fg="#4CAF50",  # Green color
    bg="#f0f0f0",
)
title_label.pack(pady=20)

# Add instructions
instructions = tk.Label(
    app,
    text="Upload a CAPTCHA image to solve it.",
    font=("Helvetica", 14),
    fg="#555555",  # Dark gray
    bg="#f0f0f0",
)
instructions.pack(pady=10)

# Upload button
upload_btn = tk.Button(
    app,
    text="Upload CAPTCHA",
    command=upload_file,
    font=("Helvetica", 14, "bold"),
    bg="#4CAF50",  # Green
    fg="white",
    activebackground="#45a049",
    activeforeground="white",
    padx=20,
    pady=10,
    relief="raised",
    borderwidth=2,
)
upload_btn.pack(pady=20)

# Display area for uploaded image
img_frame = tk.Frame(app, bg="#ffffff", relief="groove", borderwidth=2)
img_frame.pack(pady=10)

# Initially hidden image label
img_label = tk.Label(img_frame, bg="#ffffff")
# Do not pack the `img_label` initially; it will only appear after an image is uploaded

# Display prediction result or error
result_label = tk.Label(
    app,
    text="",
    font=("Helvetica", 16),
    fg="#333333",  # Darker gray
    bg="#f0f0f0",
    wraplength=500,
)
result_label.pack(pady=20)

# Run the Tkinter app
app.mainloop()
