# 🎙️ Audio Authenticity Analyzer
### Detecting AI-Generated and Manipulated Audio Using Machine Learning
*IST 584 Capstone Project*

---

A desktop application that analyzes audio recordings and produces a confidence score indicating the likelihood of AI generation or manipulation. Built with Python, it combines signal processing, machine learning, and an optional LLM-powered explanation layer — designed so that anyone can get an immediate answer, and anyone who wants to understand *why* can dig deeper.

---

## 🔍 What It Does

The tool loads an MP3 or WAV file, extracts seven acoustic features from the waveform and frequency domain, and passes them through a trained machine learning model to produce a 0–100 manipulation confidence score. The interface is built around a tiered information design:

1. **Immediate verdict** — a color-coded score bar (green / yellow / red) tells you the answer at a glance
2. **Signal evidence** — the Signal Metrics panel exposes the raw feature values that drove the score
3. **Visual inspection** — the Audio Visualizer renders the waveform and spectrogram for pattern-level review
4. **Plain-language explanation** — the Explain Results panel sends the feature data to GPT-4o-mini and returns a human-readable interpretation

This layered design means the tool works for rapid triage *and* for users who want to understand the reasoning behind an assessment — not just accept an output.

---

## 🧠 How It Works

Audio is analyzed using seven signal-processing features extracted via [Librosa](https://librosa.org/):

| Feature | What It Captures |
|---|---|
| **RMS Energy** | Average amplitude across frames |
| **Zero Crossing Rate** | Rate of waveform sign changes |
| **Spectral Centroid** | Brightness / frequency center of mass |
| **Spectral Flatness** | Tonal vs. noise-like characteristics |
| **Noise Floor Mean** | Average amplitude in silent segments |
| **Noise Floor Variance** | Natural variability in silence |
| **Near-Zero Ratio** | Proportion of frames below amplitude threshold |

Authentic recordings typically exhibit natural variability in noise and frequency content. AI-generated audio tends to appear overly smooth or produce irregular patterns in silence — particularly a suppressed or near-zero noise floor, which is one of the strongest detection signals.

Features are fed into a trained **Random Forest classifier** (scikit-learn) that outputs a probability score. If no model file is present, the system falls back to a heuristic scoring approach using the same features.

**Score thresholds:**
- 🟢 `< 40%` — Likely Authentic
- 🟡 `40–59%` — Inconclusive
- 🔴 `≥ 60%` — Likely AI-Generated or Manipulated

---

## 📊 Test Results

Tested against 12 audio samples across three categories:

| Sample | Source | Score | Result |
|---|---|---|---|
| A-01 | Authentic — iPhone recording | 14% | ✅ Likely Authentic |
| A-02 | Authentic — professional studio | 6% | ✅ Likely Authentic |
| A-03 | Authentic — informal/ambient | 1% | ✅ Likely Authentic |
| A-04 | Authentic — pro w/ noise gate & post-processing | 76% | ⚠️ False Positive |
| TTS-01 | Google WaveNet TTS — male | 98% | ✅ Detected |
| TTS-02 | Google WaveNet TTS — female | 98% | ✅ Detected |
| TTS-03 | Google WaveNet TTS — male | 96% | ✅ Detected |
| TTS-04 | Google WaveNet TTS — female | 88% | ✅ Detected |
| AI-01 | ElevenLabs neural synthesis — male | 78% | ✅ Detected |
| AI-02 | ElevenLabs neural synthesis — female | 75% | ✅ Detected |
| AI-03 | ElevenLabs neural synthesis — male | 70% | ✅ Detected |
| AI-04 | ElevenLabs neural synthesis — female | 70% | ✅ Detected |

**Overall accuracy: 91.7%** — 11 of 12 samples correctly assessed. The one false positive (A-04) was a professionally produced recording with a noise gate applied, which suppresses the natural noise floor variation the model relies on as an authenticity signal — making it acoustically indistinguishable from AI-generated silence.

**WaveNet TTS averaged 95%.** Parametric synthesis leaves strong, consistent artifacts.  
**ElevenLabs averaged 73.25%.** Deep learning-based neural synthesis narrows the detection gap but remains detectable — for now.

---

## ⚙️ Setup & Usage

### Requirements
```
Python 3.8+
librosa
numpy
scikit-learn
joblib
matplotlib
python-dotenv
openai
tkinter (included in standard Python on most platforms)
```

### Install dependencies
```bash
pip install librosa numpy scikit-learn joblib matplotlib python-dotenv openai
```

### Optional: OpenAI API key (for Explain Results feature)
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```
The tool runs fully without this — the Explain Results button will simply be disabled.

### Run
```bash
python audioauth584_app.py
```

### Usage

1. Click **Select Audio File** and choose a `.mp3` or `.wav` file
2. Read the color-coded score bar — green is likely authentic, red is likely AI-generated

### Green Score — Likely Authentic (< 40%)

---

### Yellow Score — Inconclusive (40–59%)

---

### Red Score — Likely AI-Generated (≥ 60%)

---

3. Optionally click **Signal Metrics** to see the raw feature values

### Signal Metrics Panel

---

4. Optionally click **Audio Visualizer** to inspect the waveform and spectrogram

### Audio Visualizer

---

5. Optionally click **Explain Results** for a plain-language breakdown (requires OpenAI API key)
6. Load multiple files at once — each gets its own tab; right-click a tab to close it

---

## 📁 Project Structure

```
audioauth584_app.py        # Main application — GUI and analysis pipeline
audioauth584_features.py   # Feature name definitions
model/
  audioauth584_model.pkl   # Trained Random Forest classifier
.env                       # API key (not committed — create your own)
requirements.txt           # Dependencies
README.md
```

---

## ⚠️ Limitations

- **Professional audio post-processing** (noise gates, silence normalization) can suppress the noise floor signal the model relies on, producing false positives. Best suited for raw or lightly processed recordings.
- **Global feature analysis only** — the tool measures statistics across the full recording. Localized edits or short synthetic segments embedded in otherwise authentic audio may not be detected.
- **Outputs are probabilistic**, not definitive. The tool is designed to support informed human judgment, not replace it.

---

## 🔭 What This Proves — and Where It Could Go

This project is a functional proof of concept. It demonstrates that lightweight, signal-based audio authenticity detection is achievable without specialized forensic infrastructure — and that it can be made accessible and interpretable for non-technical users. A more fully developed version, trained on a broader and more diverse dataset and extended with temporal analysis and additional acoustic features like MFCCs, would be a significantly more powerful tool.

The detection margin between WaveNet (95% avg) and ElevenLabs (73% avg) already illustrates the trajectory: as neural synthesis improves, signal-based detection becomes harder. Building robust detection tools now, while the artifacts are still measurable, is the right time to invest.

---

## 🛠️ Built With

- [Python](https://python.org)
- [Librosa](https://librosa.org/) — audio feature extraction
- [scikit-learn](https://scikit-learn.org/) — machine learning model
- [Matplotlib](https://matplotlib.org/) — waveform and spectrogram visualization
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — graphical interface
- [OpenAI API](https://platform.openai.com/) — natural language explanation (optional)
