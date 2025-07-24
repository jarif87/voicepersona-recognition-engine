# VoicePersona Recognition Engine

The **VoicePersona Recognition Engine** is a FastAPI-based web application that predicts gender ("Male" or "Female") from voice features using a pre-trained Random Forest model. The app features a modern, user-friendly interface with rounded text input fields for five key features: `meanfun`, `IQR`, `Q25`, `sd`, and `sp.ent`. It is designed with a unique purple-teal gradient background and a sleek, responsive layout.

## Features
- **Input Form**: Text inputs for voice features with HTML5 validation for the following ranges:
  - `meanfun`: 0.055565–0.237636
  - `IQR`: 0.014558–0.252225
  - `Q25`: 0.000229–0.247347
  - `sd`: 0.018363–0.115273
  - `sp.ent`: 0.738651–0.981997
- **Prediction**: Displays only the predicted gender ("Male" or "Female") in a rounded, green success box.
- **Design**: Modern UI with rounded inputs, a gradient background (purple to teal), glassmorphism effects, and hover animations.
- **Model**: Uses a pre-trained Random Forest Classifier (`model.pkl`) trained on raw features without standardization.

## Project Structure
```
VoicePersona-Recognition-Engine/
├── model/
│   └── model.pkl        # Pre-trained Random Forest model
├── static/
│   └── style.css        # CSS for styling the form
├── templates/
│   └── index.html       # HTML form for input and prediction
├── app.py               # FastAPI application
├── README.md            # This file

```

## Prerequisites
- Python 3.8+
- Required packages:
  ```
  pip install fastapi uvicorn jinja2 python-multipart pandas scikit-learn
  ```
## Setup Instructions
- Clone or Set Up the Project:
    - Create the project folder structure as shown above.
    - Ensure model.pkl is in the model folder. If not, train and save the model (see "Training the Model" below).

1. **Install Dependencies:**
- Run the following command to install required packages:bash
```
fastapi==0.115.13
catboost==1.2.8
numpy==1.26.4
uvicorn
Jinja2==3.1.6
python-multipart
```
2. **Run the Application:**
- Locally, navigate to the project directory and run:bash
```
uvicorn main:app 

```
## Usage
- Open the app in your browser (http://127.0.0.1:8000).
- Enter values for meanfun, IQR, Q25, sd, and sp.ent within the specified ranges.
- Click "Predict Gender" to see the predicted gender ("Male" or "Female").
- The result appears in a green, rounded box below the form.

