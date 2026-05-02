import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import joblib
from dotenv import load_dotenv
from openai import OpenAI

from audioauth584_features import FEATURE_NAMES

# =============================================================================
# Setup
# =============================================================================
load_dotenv()
_openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_openai_api_key) if _openai_api_key else None

MODEL_PATH = os.path.join("model", "audioauth584_model.pkl")

tab_store = {}
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model


# =============================================================================
# Helpers
# =============================================================================
def get_current_tab_data():
    tab_id = notebook.select()
    return tab_store.get(tab_id)


def clear_ui():
    filename_label.config(text="Analyzing: None")
    score_canvas.delete("all")
    assessment_label.config(text="")
    method_label.config(text="")
    metrics_button.config(state="disabled")
    explain_button.config(state="disabled")
    visualizer_button.config(state="disabled")


# =============================================================================
# Score Bar
# =============================================================================
def update_score_bar(score):
    score_canvas.delete("all")

    width = 300
    fill_width = int((score / 100) * width)

    if score < 40:
        color = "green"
    elif score < 60:
        color = "yellow"
    else:
        color = "red"

    score_canvas.create_rectangle(0, 0, fill_width, 25, fill=color)
    score_canvas.create_rectangle(0, 0, width, 25, outline="black")
    score_canvas.create_text(width / 2, 12, text=f"{score}%", fill="black")


# =============================================================================
# Popups
# =============================================================================
def show_metrics():
    data = get_current_tab_data()
    if not data:
        return

    popup = tk.Toplevel(window)
    popup.title("Signal Metrics")
    popup.geometry("600x400")

    text = tk.Text(popup, wrap="word", font=("Courier New", 10))
    text.pack(expand=True, fill="both", padx=10, pady=10)

    metrics_text = "\n".join([f"{k}: {v}" for k, v in data["results"].items()])
    text.insert("1.0", metrics_text)
    text.config(state="disabled")


def show_visualizer():
    data = get_current_tab_data()
    if not data:
        return

    waveform = data["waveform"]
    sr = data["sr"]

    popup = tk.Toplevel(window)
    popup.title("Audio Visualizer")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    librosa.display.waveshow(waveform, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")

    S = librosa.stft(waveform)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", ax=ax[1])
    ax[1].set_title("Spectrogram")
    fig.colorbar(img, ax=ax[1])

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack()


def explain_results():
    data = get_current_tab_data()
    if not data:
        return
    if not client:
        return

    metrics_text = "\n".join([f"{k}: {v}" for k, v in data["results"].items()])

    prompt = f"""
You are an audio forensic expert.

Interpret the following extracted audio metrics in simple human language.
Explain whether the recording appears real, manipulated, or AI-generated.

Avoid technical jargon. Be concise.

Metrics:
{metrics_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You analyze forensic audio metrics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        explanation = response.choices[0].message.content
    except Exception as e:
        explanation = f"Error contacting OpenAI:\n{e}"

    popup = tk.Toplevel(window)
    popup.title("AI Explanation")
    popup.geometry("600x400")

    text = tk.Text(popup, wrap="word", font=("Courier New", 11))
    text.pack(expand=True, fill="both", padx=10, pady=10)
    text.insert("1.0", explanation)
    text.config(state="disabled")


# =============================================================================
# Audio Analysis
# =============================================================================
def analyze_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
    if not file_path:
        return

    filename = os.path.basename(file_path)

    waveform, sr = librosa.load(file_path, sr=None)

    rms_frames = librosa.feature.rms(y=waveform)[0]
    avg_rms = float(np.mean(rms_frames))
    avg_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)[0]))
    avg_centroid = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]))
    avg_flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)[0]))

    threshold = np.percentile(rms_frames, 20)
    noise_frames = rms_frames[rms_frames <= threshold]
    noise_floor_mean = float(np.mean(noise_frames))
    noise_floor_var = float(np.var(noise_frames))
    near_zero_ratio = float(np.sum(rms_frames < 1e-4) / len(rms_frames))

    feature_vector = np.array([[avg_rms, avg_zcr, avg_centroid, avg_flatness,
                                noise_floor_mean, noise_floor_var, near_zero_ratio]])

    model = _load_model()

    if model:
        prob_fake = float(model.predict_proba(feature_vector)[0][1])
        confidence = int(round(prob_fake * 100))
        score_method = "ML Model (audioauth584)"
    else:
        confidence = 50
        score_method = "Heuristic"

    interpretation = (
        "Likely AI-Generated or Manipulated"
        if confidence >= 60 else
        "Likely Authentic Recording"
    )

    results = {
        "avg_rms": avg_rms,
        "avg_zcr": avg_zcr,
        "avg_centroid": avg_centroid,
        "avg_flatness": avg_flatness,
        "noise_floor_mean": noise_floor_mean,
        "noise_floor_variance": noise_floor_var,
        "near_zero_ratio": near_zero_ratio,
        "confidence_score": confidence,
        "score_method": score_method,
        "interpretation": interpretation,
    }

    # Create tab
    tab = tk.Frame(notebook)
    notebook.add(tab, text=filename[:20])
    notebook.select(tab)

    tab_store[str(tab)] = {
        "filename": filename,
        "results": results,
        "waveform": waveform,
        "sr": sr
    }

    update_ui_from_tab()


# =============================================================================
# Tab Change
# =============================================================================
def update_ui_from_tab(event=None):
    data = get_current_tab_data()
    if not data:
        clear_ui()
        return

    filename_label.config(text=f"Analyzing: {data['filename']}")
    update_score_bar(data["results"]["confidence_score"])
    assessment_label.config(text=f"Assessment: {data['results']['interpretation']}")
    method_label.config(text=f"Score Method: {data['results']['score_method']}")

    metrics_button.config(state="normal")
    explain_button.config(state="normal" if client else "disabled")  # ← changed
    visualizer_button.config(state="normal")


# =============================================================================
# Tab Close (Right Click)
# =============================================================================
def close_tab(event):
    try:
        index = notebook.index(f"@{event.x},{event.y}")
        tab_id = notebook.tabs()[index]

        notebook.forget(index)
        tab_store.pop(tab_id, None)

        if not notebook.tabs():
            clear_ui()
    except:
        pass


# =============================================================================
# GUI
# =============================================================================
window = tk.Tk()
window.title("Audio Authenticity Analyzer")
window.geometry("520x520")

notebook = ttk.Notebook(window)
notebook.pack(fill="x")
notebook.bind("<<NotebookTabChanged>>", update_ui_from_tab)
notebook.bind("<Button-3>", close_tab)

tk.Label(window, text="Audio Authenticity Analyzer",
         font=("Courier New", 16)).pack(pady=10)

filename_label = tk.Label(window, text="Analyzing: None",
                          font=("Courier New", 10), fg="steelblue")
filename_label.pack(pady=5)

select_button = tk.Button(window, text="Select Audio File",
                          font=("Courier New", 11),
                          command=analyze_audio, width=25)
select_button.pack(pady=5)

metrics_button = tk.Button(window, text="Signal Metrics",
                           font=("Courier New", 11),
                           command=show_metrics, width=25, state="disabled")
metrics_button.pack(pady=2)

explain_button = tk.Button(window, text="Explain Results",
                           font=("Courier New", 11),
                           command=explain_results, width=25, state="disabled")
explain_button.pack(pady=2)

visualizer_button = tk.Button(window, text="Audio Visualizer",
                              font=("Courier New", 11),
                              command=show_visualizer, width=25, state="disabled")
visualizer_button.pack(pady=2)

tk.Label(window, text="").pack()

tk.Label(window, text="Manipulation Detection Score",
         font=("Courier New", 12)).pack()

score_canvas = tk.Canvas(window, width=300, height=25)
score_canvas.pack(pady=5)

assessment_label = tk.Label(window, text="",
                            font=("Courier New", 11))
assessment_label.pack(pady=5)

method_label = tk.Label(window, text="",
                        font=("Courier New", 9), fg="gray")
method_label.pack(pady=10)

window.mainloop()