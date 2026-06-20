import os
# Bu gizleme komutu KESİNLİKLE 'import pygame' satırından üstte olmalıdır!
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import speech_recognition as sr
from faster_whisper import WhisperModel
from gtts import gTTS
import tempfile
import time

class AudioManager:
    def __init__(self):
        print("[SYSTEM] Loading models, please wait...")
        # 1. İYİLEŞTİRME: base.en yerine small.en kullanıyoruz. 
        # (İlk çalıştırmada modeli indireceği için biraz bekletebilir)
        self.stt_model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
        
        self.recognizer = sr.Recognizer()
        
        # 2. İYİLEŞTİRME: Mikrofon ayarları
        self.recognizer.pause_threshold = 1.0  # Konuşurken 1 saniye dursan bile kaydı kesmez (Varsayılan 0.8)
        self.recognizer.energy_threshold = 300 # Ortam sesini daha iyi izole etmesi için başlangıç eşiği
        self.recognizer.dynamic_energy_threshold = True
        
        self.audio_ready = True
        try:
            pygame.mixer.init()
        except Exception as e:
            self.audio_ready = False
            print(f"[SYSTEM] Audio output init failed: {e}")

    def speak(self, text):
        print(f"🤖 Agent: {text}")
        if not self.audio_ready:
            return

        temp_mp3_path = None
        try:
            tts = gTTS(text=text, lang='en', tld='com')
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                temp_mp3_path = temp_mp3.name

            tts.save(temp_mp3_path)
            pygame.mixer.music.load(temp_mp3_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except ConnectionError:
            print("[AUDIO WARNING] Internet unavailable - skipping TTS audio playback")
        except Exception as e:
            print(f"[AUDIO ERROR]: {e}")
        finally:
            if temp_mp3_path and os.path.exists(temp_mp3_path):
                # Ensure mixer released file handle on Windows.
                for _ in range(10):
                    try:
                        os.remove(temp_mp3_path)
                        break
                    except PermissionError:
                        time.sleep(0.05)

    def listen_and_transcribe(self):
        temp_audio_path = None
        try:
            with sr.Microphone() as source:
                print("🎤 Listening...")
                # Ortam gürültüsüne alışması için 0.5 saniye çok iyi, kalsın.
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # timeout=5 (5 saniye hiç ses gelmezse iptal et), phrase_time_limit=10 (maksimum 10 sn dinle)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

            print("[SYSTEM] Audio captured, transcribing...")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio.get_wav_data())
                temp_audio_path = temp_audio.name

            # 3. İYİLEŞTİRME: VAD Filtresi ve Beam Size eklendi!
            segments, _ = self.stt_model.transcribe(
                temp_audio_path, 
                language="en",
                beam_size=5,        # Daha derin analiz yapar, doğruluğu artırır.
                vad_filter=True,    # Sadece insan sesi olan yerleri çevirir, gürültüyü atlar.
                vad_parameters=dict(min_silence_duration_ms=500) # Yarım saniyelik boşlukları sessizlik sayar.
            )
            
            # Filter out filler words and transcribe
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            print(f"[WHISPER] Transcribed text: '{transcript}'")  # Debug logging
            return transcript.strip()

        except sr.WaitTimeoutError:
            print("[SYSTEM] No speech detected (timeout)")
            return ""
        except OSError as e:
            print(f"[ERROR] Microphone unavailable: {e}")
            return ""
        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")
            return ""
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)