import os
import cv2
import numpy as np

from audio_manager import AudioManager
from agent import CommandAgent
from video_processor import VideoProcessor
from operations import (
    operation_channels,
    operation_detect_edges,
    operation_flip,
    operation_gaussian_blur,
    operation_grayscale,
    operation_inverse,
    operation_laplacian_sharpener,
    operation_shift,
    operation_first_derivative,
    operation_second_derivative,
    operation_fourier_transform,
    operation_inverse_fourier,
)

IMAGE_PATH = os.getenv("IMAGE_PATH", "lenna.jpeg")
MAX_SHIFT_PIXELS = 2000
MAX_PERCENTAGE = 300
MAX_INTENSITY = 99
WINDOW_NAME = "ImageProcessingVoiceAgent"


class ImageProcessor:
    def __init__(self, image_path: str):
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.original = original
        self.current = original.copy()
        self.history = []  # Stack of previous states for undo
        self.history_spectrum = []  # Stack of fourier spectrums for undo
        self.last_fourier_spectrum = None

    def save_state(self, invalidate_fourier: bool = True):
        """Save current image state to history (before applying operation).
        
        Args:
            invalidate_fourier: If True, clear fourier spectrum when saving.
                              If False, preserve spectrum (e.g., for fourier operations).
        """
        self.history.append(self.current.copy())
        # Also save the current spectrum state for proper undo
        self.history_spectrum.append(self.last_fourier_spectrum)
        if invalidate_fourier:
            self.last_fourier_spectrum = None

    def undo(self) -> bool:
        """Undo last operation. Returns True if successful, False if nothing to undo."""
        if not self.history:
            return False
        self.current = self.history.pop()
        # Restore the spectrum state when undoing
        self.last_fourier_spectrum = self.history_spectrum.pop() if self.history_spectrum else None
        return True

    def reset(self):
        """Reset image to original state and clear history."""
        self.current = self.original.copy()
        self.history = []
        self.history_spectrum = []
        self.last_fourier_spectrum = None

    def show(self):
        cv2.imshow(WINDOW_NAME, self.current)
        cv2.waitKey(1)

    def shift(self, direction: str, pixels: int):
        self.save_state()
        direction_map = {"up": "n", "down": "s", "left": "w", "right": "e"}
        self.current = operation_shift(self.current, pixels, direction_map[direction])

    def change_contrast(self, percentage: int):
        """Adjust image contrast.
        
        Args:
            percentage: -100 to 100+ scale
                -100: minimum contrast (flat gray)
                0: original contrast
                100: double contrast
        """
        self.save_state()
        # Map percentage to alpha: -100→0.1, 0→1.0, 100→2.0, etc.
        alpha = 1.0 + (percentage / 100.0)
        # Clamp alpha to reasonable range [0.1, 3.0]
        alpha = max(0.1, min(alpha, 3.0))
        self.current = cv2.convertScaleAbs(self.current, alpha=alpha, beta=0)

    def change_brightness(self, percentage: int):
        """Adjust image brightness.
        
        Args:
            percentage: -100 to 100+ scale
                -100: dark (subtract 255)
                0: original
                100: bright (add 255)
        """
        self.save_state()
        # Map percentage to beta: -100→-255, 0→0, 100→255
        beta = int((percentage / 100.0) * 255.0)
        # Clamp beta to safe range [-255, 255]
        beta = max(-255, min(beta, 255))
        self.current = cv2.convertScaleAbs(self.current, alpha=1.0, beta=beta)

    def apply_blur(self, intensity: int):
        self.save_state()
        k = intensity if intensity % 2 == 1 else intensity + 1
        self.current = operation_gaussian_blur(self.current, (k, k))

    def sharpen(self, intensity: int):
        self.save_state()
        # Map 1-99 to safer amplitude range (1.0-2.5) to avoid clipping artifacts
        # This prevents harsh over-sharpening that creates visual artifacts
        amp = 1.0 + (intensity / 99.0) * 1.5
        self.current = operation_laplacian_sharpener(self.current, amplitude=amp)

    def convert_grayscale(self):
        self.save_state()
        self.current = operation_grayscale(self.current)

    def detect_edges(self):
        self.save_state()
        self.current = operation_detect_edges(self.current)

    def split_channel(self, color: str):
        self.save_state()
        color_map = {"red": "r", "green": "g", "blue": "b"}
        self.current = operation_channels(self.current, color_map[color], colored=True)

    def flip(self, axis: str):
        self.save_state()
        axis_map = {"horizontal": "h", "vertical": "v"}
        self.current = operation_flip(self.current, axis_map[axis])

    def invert(self):
        self.save_state()
        self.current = operation_inverse(self.current)

    def first_derivative(self, direction: str = "both"):
        """Apply first-order derivative using Sobel operator."""
        self.save_state()
        self.current = operation_first_derivative(self.current, direction)

    def second_derivative(self):
        """Apply second-order derivative using Laplacian operator."""
        self.save_state()
        self.current = operation_second_derivative(self.current)

    def fourier_transform(self):
        """Apply Fast Fourier Transform and display magnitude spectrum."""
        self.save_state(invalidate_fourier=False)
        self.current, self.last_fourier_spectrum = operation_fourier_transform(self.current, return_spectrum=True)

    def inverse_fourier_transform(self) -> bool:
        """Reconstruct image from the last stored Fourier spectrum."""
        if self.last_fourier_spectrum is None:
            return False
        self.save_state(invalidate_fourier=False)
        self.current = operation_inverse_fourier(self.last_fourier_spectrum)
        self.last_fourier_spectrum = None
        return True


def _valid_range(value, min_value, max_value):
    return value is not None and min_value <= value <= max_value


def safe_int(value):
    """Gelen değerin güvenli bir şekilde tam sayı olup olmadığını kontrol eder."""
    try:
        if value is None:
            return None
        return int(value)
    except (ValueError, TypeError):
        return None


def choose_mode():
    """Kullanıcıya foto vs webcam seçimi sun"""
    print("\n" + "="*60)
    print("📸 MODE SELECTION - İmage Processing Voice Agent")
    print("="*60)
    print("1️⃣  Photo Mode    - Process static image with voice commands")
    print("2️⃣  Webcam Mode   - Process live video stream with voice commands")
    print("="*60)
    
    while True:
        choice = input("Select mode (1 or 2): ").strip()
        if choice == "1":
            print("✅ Photo Mode selected\n")
            return "photo"
        elif choice == "2":
            print("✅ Webcam Mode selected\n")
            return "webcam"
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def choose_fps():
    """Webcam için hedef FPS seçimi sun"""
    print("\n" + "-"*60)
    print("🎥 WEBCAM FPS SELECTION")
    print("-"*60)
    print("1️⃣  30 FPS  - Daha akıcı, biraz daha fazla CPU")
    print("2️⃣  15 FPS  - Daha hafif, daha az CPU")
    print("-"*60)

    while True:
        choice = input("Select FPS (1 or 2): ").strip()
        if choice == "1":
            print("✅ 30 FPS selected\n")
            return 30
        elif choice == "2":
            print("✅ 15 FPS selected\n")
            return 15
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def main_webcam(fps=30):
    """Webcam mode - process live video stream with voice commands"""
    print("\n[SYSTEM] Initializing webcam mode...")
    
    try:
        video = VideoProcessor(camera_id=0, fps=fps, max_history=5)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        print("[ERROR] Please check your webcam connection")
        return
    
    audio = AudioManager()
    agent = CommandAgent()
    
    print("[SYSTEM] Starting video capture...")
    video.start_capture()
    
    audio.speak("Webcam mode activated. Ready to process video.")
    
    try:
        while True:
            print("\n" + "="*60)
            user_input = input("🟢 Press ENTER to listen (type 'q' to quit): ").strip()
            
            if user_input.lower() == 'q':
                audio.speak("Shutting down the system. Goodbye.")
                break
            
            user_text = audio.listen_and_transcribe()
            
            if not user_text:
                print("[SYSTEM] No speech detected. Try again.")
                continue
            
            print(f"\n👤 You: {user_text}")
            
            # Check for exit commands
            if "exit" in user_text.lower() or "stop" in user_text.lower():
                audio.speak("Shutting down the system. Goodbye.")
                break
            
            # Process command
            response = agent.process_command(user_text)
            action = response.get("action")
            
            # APPLY OPERATIONS TO VIDEO
            if action == "apply_blur":
                intensity = safe_int(response.get("intensity"))
                if not _valid_range(intensity, 1, MAX_INTENSITY):
                    audio.speak("How much blur? Say 1 to 99, or try 'a lot', 'slightly'.")
                else:
                    video.apply_operation("apply_blur", {"intensity": intensity})
                    audio.speak(f"Applying blur with intensity {intensity} to video stream.")
                    agent.clear_history()
            
            elif action == "sharpen_image":
                intensity = safe_int(response.get("intensity"))
                if not _valid_range(intensity, 1, MAX_INTENSITY):
                    audio.speak("How much sharpening? Say 1 to 99, or try 'a lot', 'slightly'.")
                else:
                    video.apply_operation("sharpen_image", {"intensity": intensity})
                    audio.speak(f"Sharpening video stream with intensity {intensity}.")
                    agent.clear_history()
            
            elif action == "flip_image":
                axis = response.get("axis")
                if axis not in {"horizontal", "vertical"}:
                    audio.speak("Horizontal or vertical flip?")
                else:
                    video.apply_operation("flip_image", {"axis": axis})
                    audio.speak(f"Applying {axis} flip to video.")
                    agent.clear_history()
            
            elif action == "invert_image":
                video.apply_operation("invert_image", {})
                audio.speak("Inverting video colors.")
                agent.clear_history()
            
            elif action == "convert_grayscale":
                video.apply_operation("convert_grayscale", {})
                audio.speak("Converting video to grayscale.")
                agent.clear_history()
            
            elif action == "detect_edges":
                video.apply_operation("detect_edges", {})
                audio.speak("Applying edge detection to video.")
                agent.clear_history()
            
            elif action == "change_brightness":
                percentage = safe_int(response.get("percentage"))
                if not _valid_range(percentage, -MAX_PERCENTAGE, MAX_PERCENTAGE):
                    audio.speak(f"Brightness level between -300 and 300?")
                else:
                    video.apply_operation("change_brightness", {"percentage": percentage})
                    audio.speak(f"Adjusting brightness by {percentage} percent.")
                    agent.clear_history()
            
            elif action == "change_contrast":
                percentage = safe_int(response.get("percentage"))
                if not _valid_range(percentage, -MAX_PERCENTAGE, MAX_PERCENTAGE):
                    audio.speak(f"Contrast level between -300 and 300?")
                else:
                    video.apply_operation("change_contrast", {"percentage": percentage})
                    audio.speak(f"Adjusting contrast by {percentage} percent.")
                    agent.clear_history()
            
            elif action == "shift_image":
                direction = response.get("direction")
                pixels = safe_int(response.get("pixels"))
                if direction not in {"up", "down", "left", "right"}:
                    audio.speak("Which direction? Up, down, left, or right?")
                elif not _valid_range(pixels, 1, MAX_SHIFT_PIXELS):
                    audio.speak(f"How many pixels? 1 to {MAX_SHIFT_PIXELS}?")
                else:
                    video.apply_operation("shift_image", {"direction": direction, "pixels": pixels})
                    audio.speak(f"Shifting video {direction} by {pixels} pixels.")
                    agent.clear_history()
            
            elif action == "split_channel":
                color = response.get("color")
                if color not in ["red", "green", "blue"]:
                    audio.speak("Red, green, or blue channel?")
                else:
                    video.apply_operation("split_channel", {"color": color})
                    audio.speak(f"Displaying {color} channel.")
                    agent.clear_history()
            
            elif action == "first_derivative":
                direction = response.get("direction")
                if direction not in ["x", "y", "both"]:
                    audio.speak("Which derivative? Say horizontal (x), vertical (y), or both?")
                else:
                    video.apply_operation("first_derivative", {"direction": direction})
                    audio.speak(f"Applying first-order derivative in {direction} direction to video.")
                    agent.clear_history()
            
            elif action == "second_derivative":
                video.apply_operation("second_derivative", {})
                audio.speak("Applying second-order derivative using Laplacian operator to video.")
                agent.clear_history()
            
            elif action == "fourier_transform":
                video.apply_operation("fourier_transform", {})
                audio.speak("Displaying frequency spectrum using Fast Fourier Transform on video.")
                agent.clear_history()

            elif action == "inverse_fourier_transform":
                video.apply_operation("inverse_fourier_transform", {})
                audio.speak("Applying inverse Fast Fourier Transform on video stream.")
                agent.clear_history()
            
            elif action == "undo":
                if video.undo():
                    audio.speak("Undo: reverted to previous frame.")
                else:
                    audio.speak("Nothing to undo.")
            
            elif action == "reset":
                video.reset()
                audio.speak("Reset: video restored to original.")
                agent.clear_history()
            
            else:
                # Unknown action
                if action == "unknown":
                    msg = response.get("message", "I didn't understand that.")
                    audio.speak(msg)
                else:
                    audio.speak(response.get("message", "I didn't catch that."))
    
    finally:
        print("\n[SYSTEM] Cleaning up...")
        video.stop()
        print("[SYSTEM] Webcam mode closed.")


def main():
    """Photo mode - process static image"""
    try:
        processor = ImageProcessor(IMAGE_PATH)
    except FileNotFoundError as exc:
        print(f"[SYSTEM] {exc}")
        return

    audio = AudioManager()
    agent = CommandAgent()
    
    audio.speak("System initialized. I am ready when you are.")
    processor.show()

    while True:
        print("\n" + "="*50)
        user_input = input("🟢 Dinlemeyi başlatmak için 'ENTER' tuşuna basın (Çıkmak için 'q' yaz): ")
        
        if user_input.lower() == 'q':
            audio.speak("Shutting down the system. Goodbye.")
            break

        user_text = audio.listen_and_transcribe()
        
        if not user_text:
            print("[SİSTEM] Hiçbir ses algılanmadı. Bekleme moduna dönülüyor.")
            continue

        print(f"\n👤 You: {user_text}")
        
        if "exit" in user_text.lower() or "stop" in user_text.lower():
            audio.speak("Shutting down the system. Goodbye.")
            break

        response = agent.process_command(user_text)
        action = response.get("action")

        if action == "shift_image":
            pixels = safe_int(response.get("pixels"))
            direction = response.get("direction")
            if direction not in {"up", "down", "left", "right"}:
                audio.speak("Which direction? Try up, down, left, or right.")
            elif not _valid_range(pixels, 1, MAX_SHIFT_PIXELS):
                audio.speak(f"How many pixels? Say a number from 1 to {MAX_SHIFT_PIXELS}.")
            else:
                processor.shift(direction, pixels)
                processor.show()
                audio.speak(f"Understood. Shifting the image to the {direction} by {pixels} pixels.")
                agent.clear_history()

        elif action == "flip_image":
            axis = response.get("axis")
            if axis not in {"horizontal", "vertical"}:
                audio.speak("Do you want a horizontal or vertical flip?")
            else:
                processor.flip(axis)
                processor.show()
                audio.speak(f"Understood. Applying {axis} flip.")
                agent.clear_history()

        elif action == "invert_image":
            processor.invert()
            processor.show()
            audio.speak("Understood. Inverting the image colors.")
            agent.clear_history()
                
        elif action == "change_contrast":
            percentage = safe_int(response.get("percentage"))
            if not _valid_range(percentage, -MAX_PERCENTAGE, MAX_PERCENTAGE):
                audio.speak("How much should I change the contrast? Say a number between -300 and 300.")
            else:
                processor.change_contrast(percentage)
                processor.show()
                audio.speak(f"Got it. Adjusting the contrast by {percentage} percent.")
                agent.clear_history()

        elif action == "change_brightness":
            percentage = safe_int(response.get("percentage"))
            if not _valid_range(percentage, -MAX_PERCENTAGE, MAX_PERCENTAGE):
                audio.speak("How much should I change the brightness? Say a number between -300 and 300.")
            else:
                processor.change_brightness(percentage)
                processor.show()
                audio.speak(f"Okay, changing the brightness level by {percentage} percent.")
                agent.clear_history()

        elif action == "apply_blur":
            intensity = safe_int(response.get("intensity"))
            if not _valid_range(intensity, 1, MAX_INTENSITY):
                audio.speak("How much blur do you want? Say a number from 1 to 99, or try 'slightly', 'normal', or 'a lot'.")
            else:
                processor.apply_blur(intensity)
                processor.show()
                audio.speak(f"Understood. Applying a blur filter with an intensity of {intensity}.")
                agent.clear_history()

        elif action == "sharpen_image":
            intensity = safe_int(response.get("intensity"))
            if not _valid_range(intensity, 1, MAX_INTENSITY):
                audio.speak("How much sharpening do you want? Say a number from 1 to 99, or try 'slightly', 'normal', or 'a lot'.")
            else:
                processor.sharpen(intensity)
                processor.show()
                audio.speak(f"Got it. Sharpening the image with an intensity of {intensity}.")
                agent.clear_history()

        elif action == "convert_grayscale":
            processor.convert_grayscale()
            processor.show()
            audio.speak("Understood. Converting the image to black and white.")
            agent.clear_history()

        elif action == "detect_edges":
            processor.detect_edges()
            processor.show()
            audio.speak("Okay, applying edge detection filter.")
            agent.clear_history()

        elif action == "undo":
            if processor.undo():
                processor.show()
                audio.speak("Undo: reverted to previous state.")
            else:
                audio.speak("Nothing to undo.")

        elif action == "reset":
            processor.reset()
            processor.show()
            audio.speak("Reset: image restored to original state.")
            agent.clear_history()

        elif action == "split_channel":
            color = response.get("color")
            if color not in ["red", "green", "blue"]:
                audio.speak("Which color channel? Say red, green, or blue.")
            else:
                processor.split_channel(color)
                processor.show()
                audio.speak(f"Understood. Displaying only the {color} color channel.")
                agent.clear_history()

        elif action == "first_derivative":
            direction = response.get("direction")
            if direction not in ["x", "y", "both"]:
                audio.speak("Which derivative? Say horizontal (x), vertical (y), or both?")
            else:
                processor.first_derivative(direction)
                processor.show()
                audio.speak(f"Understood. Applying first-order derivative in {direction} direction.")
                agent.clear_history()

        elif action == "second_derivative":
            processor.second_derivative()
            processor.show()
            audio.speak("Understood. Applying second-order derivative using Laplacian operator.")
            agent.clear_history()

        elif action == "fourier_transform":
            processor.fourier_transform()
            processor.show()
            audio.speak("Understood. Displaying frequency spectrum using Fast Fourier Transform.")
            agent.clear_history()

        elif action == "inverse_fourier_transform":
            if processor.inverse_fourier_transform():
                processor.show()
                audio.speak("Understood. Reconstructing image from the latest Fourier spectrum.")
                agent.clear_history()
            else:
                audio.speak("I need a Fourier transform first. Say: apply fourier transform.")

        else:
            # Catch-all for unknown or unclear commands
            if action == "unknown":
                msg = response.get("message", "I didn't understand that. Please be more specific.")
                audio.speak(msg)
            else:
                audio.speak(response.get("message", "I didn't quite catch that command. Can you repeat?"))
            
            # Don't clear history for unknown - user might follow up

    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = choose_mode()
    
    if mode == "photo":
        main()
    else:
        fps = choose_fps()
        main_webcam(fps=fps)