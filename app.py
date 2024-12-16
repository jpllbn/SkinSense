from pathlib import Path
from tkinter import Tk, Canvas, Button, Label, PhotoImage
import pyglet, os
import tensorflow as tf
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

pyglet.font.add_file('./assets/Fonts/Poppins/Poppins-Bold.ttf')

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\USER\Documents\SkinSense\assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Load the trained model
MODEL_PATH = 'skin_types.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_labels = ['Normal', 'Oily', 'Dry']

# Tkinter window
window = Tk()
window.title("SkinSense")
window.geometry("476x316")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=316,
    width=476,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)

image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(238.0, 159.0, image=image_image_1)

canvas.create_text(
    13.0,
    0.0,
    anchor="nw",
    text="SkinSense",
    fill="#0A3981",
    font=("Poppins", 36 * -1, "bold")
)

# Placeholder for image_2, will be replaced by uploaded image after selection
image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(181.0, 163.0, image=image_image_2)

# Accuracy label for displaying predictions
accuracy_label = Label(window, text="Skin Type: ", font=("Poppins", 8, "bold"), bg="#BCEBFF")
accuracy_label.place(x=306, y=228)

confidence_label = Label(window, text="Confidence: ", font=("Poppins", 8, "bold"), bg="#BCEBFF")
confidence_label.place(x=306, y=250)

# Image upload function
def upload_image_and_predict():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Load and preprocess the uploaded image
        frame = cv2.imread(file_path)
        if frame is None:
            print("Error: Failed to load the image.")
            return

        resized_frame = cv2.resize(frame, (128, 128))  # Resize to model input size
        normalized_frame = resized_frame / 255.0  # Normalize
        input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(input_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        confidence_score = np.max(predictions) * 100  # Get the highest confidence score in percentage

        # Update the accuracy and confidence labels in the UI
        accuracy_label.config(text=f"Skin Type: {predicted_label}")
        confidence_label.config(text=f"Confidence: {confidence_score:.2f}%")

        # Display the uploaded image in the UI
        img = Image.open(file_path)
        img = img.resize((150, 150))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)

        # Update image_2 on the canvas with the uploaded image
        canvas.itemconfig(image_2, image=img_tk)
        canvas.image = img_tk

# Real-time video capture and prediction function
def real_time_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for prediction
        resized_frame = cv2.resize(frame, (128, 128))  # Resize to model input size
        normalized_frame = resized_frame / 255.0  # Normalize
        input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(input_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Update the accuracy label in Tkinter UI
        accuracy_label.config(text=f"Predicted Skin Type: {predicted_label}")

        # Overlay the predicted label on the video stream using OpenCV
        font = cv2.QT_FONT_NORMAL
        cv2.putText(frame, f"Skin Type: {predicted_label}", (10, 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)

        # Show the frame with the overlay
        cv2.imshow("SkinSense", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# Buttons for real-time detection and image upload
button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    bg="#BCEBFF",
    activebackground="#BCEBFF",
    relief="flat",
    command=upload_image_and_predict
)
button_1.place(x=306.0, y=115.0, width=136.0, height=40.0)

button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    bg="#BCEBFF",
    activebackground="#BCEBFF",
    relief="flat",
    command=real_time_detection
)
button_2.place(x=306.0, y=164.0, width=112.0, height=40.0)

window.resizable(False, False)
window.mainloop()
