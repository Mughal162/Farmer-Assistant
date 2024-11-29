import cohere
import pyttsx3
import speech_recognition as sr
import tkinter as tk
from tkinter import Text, filedialog, Button, Label
from PIL import Image, ImageTk
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import json
import requests



cohere_api_key = 'JgjOSTltw3yNScKVY15VE5ztSndgovlumaLTe028'
co = cohere.Client(cohere_api_key)


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        display_message("Recognizing...","Assistant")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}")
        if "over" in query.lower():
            return "over"
        return query.lower()
    
    except sr.RequestError as e:
        print(f"Couldn't request results; {e}")
        speak("Couldn't reach recognition services, check your internet connection...")
        return "None"
    
    except sr.UnknownValueError as e:
        print(f"Couldn't Understand; {e}")
        speak("Couldn't understand that, Please type...")
        return "None"

# MobileNet Model
model_image = tf.keras.models.load_model('MobileNet\\model_20240815-153554.h5')

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)


def predict_crop(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the crop
    predictions = model_image.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name

def generate_context_of_image(crop_name):
    try:
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=f"Provide a small context for the : {crop_name}",
            max_tokens=300
        )
        context_image = response.generations[0].text.strip()
        entry_answer.config(state=tk.NORMAL)
        display_message(f"{context_image}","Assistant")
        entry_answer.config(state=tk.DISABLED)
    except Exception as e:
        print(f"Error generating context: {e}")
        return "None"


def select_image():
    img_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", ".jpg;.jpeg;*.png")])
    
    if img_path:
        crop_name = predict_crop(img_path)
        img = Image.open(img_path)
        img = img.resize((120,120), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        result_label.config(text=f'The crop is: {crop_name}', image=photo, compound='right')
        result_label.image=photo
        generate_context_of_image(crop_name)
    else:
        result_label.config(text="No image selected.")


# T5 model
model_path = './t5_demo'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


def generate_context_with_cohere(question):
    try:
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=f"Provide a detailed context for the question: {question}",
            max_tokens=300
        )
        context = response.generations[0].text.strip()
        return context
    except Exception as e:
        print(f"Error generating context: {e}")
        return ""

def translate_to_english(query):
    try:
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=f"Translate this to English: {query}",
            max_tokens=100
        )
        translation = response.generations[0].text.strip()
        return translation
    except Exception as e:
        print(f"Error translating query: {e}")
        return query

def answer_question(question):
    context = generate_context_with_cohere(question)
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True, padding="max_length")
    output = model.generate(
        inputs['input_ids'],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def handle_query(query):
    if query != "None":
        entry_answer.config(state=tk.NORMAL)
        translated_query = translate_to_english(query)
        if use_cohere:
            model_answer = None  
            corrected_answer = co.generate(
                model='command-xlarge-nightly',
                prompt=translated_query,
                max_tokens=300
            ).generations[0].text.strip()
            display_message(f"Assistant: {corrected_answer}", "Assistant")
        else:
            model_answer = answer_question(translated_query)
            display_message(f"Assistant: {model_answer}", "Assistant")
        entry_answer.config(state=tk.DISABLED)
    else:
        print("No query received...")

def display_message(message, sender):
    if sender == "User":
        entry_answer.insert(tk.END, f"{message}\n\n", "user")
    else:
        entry_answer.insert(tk.END, f"{message}\n\n", "assistant")

def on_type():
    entry_answer.config(state=tk.NORMAL)
    query = entry_query.get("1.0", tk.END).strip().lower()
    if query:
        display_message(f"User: {query}", "User")
        handle_query(query)
        entry_query.delete("1.0", tk.END)
        entry_answer.config(state=tk.DISABLED)

def on_speak():
    global listening
    entry_answer.config(state=tk.NORMAL)
    display_message("Listening...", "Assistant")
    listening = True
    query = takeCommand().lower()
    if query != "over":
        if listening:
            display_message(f"User: {query}", "User")
            handle_query(query)
        else:
            display_message("Listening stopped.", "Assistant")
    else:
        query = query.replace("over", "").strip()
        if query:
            display_message(f"User: {query}", "User")
            handle_query(query)
            entry_answer.config(state=tk.DISABLED)


def switch_ai():
    global use_cohere
    use_cohere = not use_cohere
    if use_cohere:
        btn_switch_ai.config(text="Switch-Model")
    else:
        btn_switch_ai.config(text="Switch-AI")
    entry_answer.config(state=tk.NORMAL)
    status = "Using Cohere" if use_cohere else "Using Model"
    display_message(f"Switched to {status} mode.", "Assistant")
    entry_answer.config(state=tk.DISABLED)

def speak_answer():
    answer_text = entry_answer.get("1.0", tk.END).strip().split('\n\n')
    latest_entry = answer_text[-1] 
    speak(latest_entry)

def clear_screen():
    entry_answer.config(state=tk.NORMAL)
    entry_answer.delete("1.0", tk.END)
    entry_answer.config(state=tk.DISABLED)


def fetch_news(api_key):
    url = f"https://newsapi.org/v2/everything?q=agriculture+India+Pakistan&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    return data.get('articles', [])

# Function to update the news labels
def update_news():

    for widget in news_frame.winfo_children():
        widget.destroy()
    news = fetch_news(api_key="b99c5b9d3ddd46ddbe5d642c4529e1e5")
    
    if not news:
        error_label = tk.Label(news_frame, text="No news found or failed to fetch news.", fg='red')
        error_label.pack(anchor='w', padx=10)
        return
    
    for i, article in enumerate(news[:5]):
        title = article.get('title', 'No Title Available')
        url = article.get('url', '#')
        
        title_label = tk.Label(news_frame, text=title, wraplength=500, justify='left', bg='#1e1e1e', fg='red', cursor='hand2')
        title_label.pack(anchor='w', padx=10)
        
        title_label.bind("<Button-1>", lambda e, url=url: open_link(url))
        labels.append(title_label)

def blink_news():
    for label in labels:
        current_color = label.cget("fg")
        next_color = "red" if current_color == "white" else "white"
        label.config(fg=next_color)
    news_frame.after(500, blink_news)
# Function to open a link in the browser
def open_link(url):
    import webbrowser
    webbrowser.open(url)

# Main window
root = tk.Tk()
root.title("Farmer Assistant")
root.configure(bg='#1e1e1e')  


frame_main = tk.Frame(root, bg='#1e1e1e')
frame_main.pack(pady=10, fill=tk.BOTH, expand=True)

# canvas = tk.Canvas(frame_main)
# canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
# scrollbar = tk.Scrollbar(frame_main, orient=tk.VERTICAL, command=canvas.yview)
# scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
# canvas.configure(yscrollcommand=scrollbar.set)
# canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
# scrollable_frame = ttk.Frame(canvas)
# canvas.create_window((0,0), window=scrollable_frame, anchor="nw")

entry_answer = Text(frame_main, height=20, width=80, state=tk.DISABLED, wrap=tk.WORD, bg='white', fg='#ffffff', insertbackground='#ffffff')
entry_answer.pack(pady=10, fill=tk.BOTH, expand=True)
entry_answer.tag_config("user", foreground="black", justify='right', font=("Helvetica", 18))
entry_answer.tag_config("assistant", foreground="black", justify='left', font=("Helvetica", 18))

frame_query = tk.Frame(frame_main, bg='#1e1e1e')
frame_query.pack()

entry_query = Text(frame_query, height=2, width=50, bg='#2e2e2e', fg='#ffffff', insertbackground='#ffffff')
entry_query.pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)

btn_enter = tk.Button(frame_query, text="Enter", command=on_type, bg='#4caf50', fg='#ffffff', bd=0, padx=10, pady=5, font=("Helvetica", 10, "bold"))
btn_enter.pack(side=tk.LEFT, padx=5)

frame_options = tk.Frame(frame_main, bg='#1e1e1e')
frame_options.pack(pady=10)

btn_clear = tk.Button(frame_options, text="Clear", command=clear_screen, bg='#f44336', fg='#ffffff', bd=0, padx=10, pady=5, font=("Helvetica", 10, "bold"))
btn_clear.pack(side=tk.LEFT, padx=5)

btn_speak_answer = tk.Button(frame_options, text="Speak", command=speak_answer, bg='#2196f3', fg='#ffffff', bd=0, padx=10, pady=5, font=("Helvetica", 10, "bold"))
btn_speak_answer.pack(side=tk.LEFT, padx=5)

btn_switch_ai = tk.Button(frame_options, text="Switch-AI", command=switch_ai, bg='#ff9800', fg='#ffffff', bd=0, padx=10, pady=5, font=("Helvetica", 10, "bold"))
btn_switch_ai.pack(side=tk.LEFT, padx=5)

mic_icon = Image.open("mic.png")  
mic_icon = mic_icon.resize((20, 20), Image.LANCZOS)
mic_img = ImageTk.PhotoImage(mic_icon)

btn_speak = tk.Button(frame_query, image=mic_img, command=on_speak, bg='#ff9800', fg='#ffffff', bd=0, highlightthickness=0)
btn_speak.pack(side=tk.LEFT, padx=5)

camera_icon = Image.open("camera_icon.png")
camera_icon = camera_icon.resize((40, 40),  Image.LANCZOS)
camera_icon = ImageTk.PhotoImage(camera_icon)

upload_button = Button(root, image=camera_icon, command=select_image, bg='#1e1e1e',  bd=0, highlightthickness=0)
upload_button.pack(pady=20)

result_label = Label(root, text="", font=("Helvetica", 16), bg='#1e1e1e', fg='lightgreen')
result_label.pack(pady=20,fill=tk.BOTH, expand=True)

news_frame = tk.Frame(root, bg='#1e1e1e')
news_frame.pack(pady=10, fill=tk.X, side=tk.BOTTOM)

labels = []
update_news()
blink_news()

use_cohere = False  
listening = False

root.mainloop()