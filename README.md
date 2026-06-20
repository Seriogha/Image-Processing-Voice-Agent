# 🎙️ Voice-Controlled Computer Vision Agent

A real-time AI assistant that processes live video streams and responds to voice commands using a fully multithreaded architecture.

## 🚀 Overview
This project integrates a local Large Language Model (Llama 3.1) and natural language intent parsing (Faster-Whisper) with real-time computer vision (OpenCV). To prevent system bottlenecks and video feed lag during heavy digital signal processing operations, the application is built on an asynchronous and multithreaded foundation.

## ⚙️ Features
* **Live Video Processing:** Real-time visual analysis using OpenCV.
* **Voice Command Recognition:** Local, fast voice transcription via Faster-Whisper.
* **Local LLM Integration:** Intent processing and response generation using Llama 3.1.
* **Asynchronous Execution:** Multithreaded pipeline ensuring a zero-lag video feed even during heavy computational tasks.

## 🛠️ Technologies & Tools
* **Language:** Python
* **Computer Vision:** OpenCV
* **AI & NLP:** Llama 3.1, Faster-Whisper
