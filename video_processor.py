import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
from operations import (
    operation_gaussian_blur,
    operation_laplacian_sharpener,
    operation_flip,
    operation_grayscale,
    operation_detect_edges,
    operation_inverse,
    operation_shift,
    operation_channels,
    operation_first_derivative,
    operation_second_derivative,
    operation_fourier_transform,
    operation_inverse_fourier,
)


class VideoProcessor:
    """Real-time video processor with threading and frame history management"""
    
    def __init__(self, camera_id=0, fps=30, max_history=5, mode_name="Webcam", max_shift_pixels=2000):
        """
        Initialize VideoProcessor
        
        Args:
            camera_id: Camera index (0 = default webcam)
            fps: Target frames per second
            max_history: Number of frames to keep for undo
        """
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"[ERROR] Cannot open camera {camera_id}")
        
        # Get actual camera resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        self.frame_delay = max(1, int(1000 / fps))  # milliseconds
        self.mode_name = mode_name
        self.max_shift_pixels = max_shift_pixels
        
        print(f"[VIDEO] Camera initialized: {self.width}x{self.height} @ {self.fps} FPS")
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=2)
        self.operation_queue = queue.Queue()
        
        # Frame history for undo (with lock for thread safety)
        self.frame_history = deque(maxlen=max_history)
        self.current_frame = None
        self.original_frame = None
        self.last_fourier_spectrum = None
        self.active_operations = []
        self.frame_lock = threading.Lock()
        
        # Thread management
        self.is_running = False
        self.capture_thread = None
        self.processing_thread = None
        
        # Statistics
        self.frame_count = 0
        self.last_operation = None
        self.last_display_time = None
        self.display_fps = 0.0
    
    def start_capture(self):
        """Start video capture in background threads"""
        print("[VIDEO] Starting capture threads...")
        self.is_running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        
        self.capture_thread.start()
        self.processing_thread.start()
        print("[VIDEO] Threads started")
    
    def _capture_loop(self):
        """Continuous video capture from camera"""
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("[VIDEO] Camera read failed")
                break
            
            # Store current frame with thread safety
            with self.frame_lock:
                self.current_frame = frame
                if self.original_frame is None:
                    self.original_frame = frame.copy()
            
            self.frame_count += 1
            
            # Add to queue if not full
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue full
    
    def _processing_loop(self):
        """Process frames with pending operations"""
        while self.is_running:
            if self.current_frame is None:
                time.sleep(0.01)
                continue
            
            # Check for pending operations
            try:
                operation = self.operation_queue.get(timeout=0.01)
                with self.frame_lock:
                    self.frame_history.append(operation)
                    self.active_operations.append(operation)
                    self.last_operation = operation.get("action")
                    print(f"[VIDEO] Operation activated: {self.last_operation}")
            except queue.Empty:
                pass
            
            # Display current frame
            self._display_frame()
    
    def _apply_operation(self, frame, operation):
        """Apply operation to single frame"""
        if frame is None:
            return None
        
        action = operation.get("action")
        
        try:
            if action == "apply_blur":
                intensity = operation.get("intensity", 5)
                k = intensity if intensity % 2 == 1 else intensity + 1
                k = max(1, min(k, 99))  # Clamp kernel size
                return operation_gaussian_blur(frame, (k, k))
            
            elif action == "sharpen_image":
                intensity = operation.get("intensity", 10)
                # Map 1-99 to amplitude 1.0-2.5
                amp = 1.0 + (intensity / 99.0) * 1.5
                return operation_laplacian_sharpener(frame, amplitude=amp)
            
            elif action == "flip_image":
                axis = operation.get("axis")
                if axis == "horizontal":
                    return operation_flip(frame, 'h')
                elif axis == "vertical":
                    return operation_flip(frame, 'v')
                return frame
            
            elif action == "convert_grayscale":
                return operation_grayscale(frame)
            
            elif action == "detect_edges":
                return operation_detect_edges(frame)
            
            elif action == "invert_image":
                return operation_inverse(frame)
            
            elif action == "shift_image":
                direction = operation.get("direction")
                pixels = operation.get("pixels", 10)
                pixels = max(0, min(pixels, self.max_shift_pixels))
                
                direction_map = {
                    "up": "n",
                    "down": "s", 
                    "left": "w",
                    "right": "e"
                }
                dir_code = direction_map.get(direction, "n")
                return operation_shift(frame, pixels, dir_code)
            
            elif action == "split_channel":
                color = operation.get("color")
                if color in ["red", "green", "blue"]:
                    color_map = {"red": "r", "green": "g", "blue": "b"}
                    return operation_channels(frame, color_map[color], colored=True)
                return frame
            
            elif action == "change_brightness":
                percentage = operation.get("percentage", 0)
                # Map percentage to beta: -100→-255, 0→0, 100→255
                beta = int((percentage / 100.0) * 255.0)
                # Clamp beta to safe range [-255, 255]
                beta = max(-255, min(beta, 255))
                return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
            
            elif action == "change_contrast":
                percentage = operation.get("percentage", 0)
                # Map percentage to alpha: -100→0.1, 0→1.0, 100→2.0, etc.
                alpha = 1.0 + (percentage / 100.0)
                # Clamp alpha to reasonable range [0.1, 3.0]
                alpha = max(0.1, min(alpha, 3.0))
                return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
            
            elif action == "first_derivative":
                direction = operation.get("direction", "both")
                return operation_first_derivative(frame, direction)
            
            elif action == "second_derivative":
                return operation_second_derivative(frame)
            
            elif action == "fourier_transform":
                spectrum_img, spectrum = operation_fourier_transform(frame, return_spectrum=True)
                self.last_fourier_spectrum = spectrum
                return spectrum_img

            elif action == "inverse_fourier_transform":
                if self.last_fourier_spectrum is None:
                    _, spectrum = operation_fourier_transform(frame, return_spectrum=True)
                    return operation_inverse_fourier(spectrum)
                reconstructed = operation_inverse_fourier(self.last_fourier_spectrum)
                self.last_fourier_spectrum = None
                return reconstructed
            
            else:
                print(f"[VIDEO] Unknown action: {action}")
                return frame
        
        except Exception as e:
            print(f"[VIDEO ERROR] Operation failed: {action} - {e}")
            return frame

    def _format_operation_summary(self, operation):
        """Create a compact label for a queued webcam operation"""
        action = operation.get("action", "unknown")

        if action == "apply_blur":
            return f"blur({operation.get('intensity', '?')})"
        if action == "sharpen_image":
            return f"sharpen({operation.get('intensity', '?')})"
        if action == "flip_image":
            return f"flip({operation.get('axis', '?')})"
        if action == "shift_image":
            return f"shift({operation.get('direction', '?')},{operation.get('pixels', '?')})"
        if action == "change_brightness":
            return f"brightness({operation.get('percentage', '?')})"
        if action == "change_contrast":
            return f"contrast({operation.get('percentage', '?')})"
        if action == "split_channel":
            return f"channel({operation.get('color', '?')})"
        if action == "convert_grayscale":
            return "grayscale"
        if action == "detect_edges":
            return "edges"
        if action == "invert_image":
            return "invert"
        if action == "first_derivative":
            return f"deriv1({operation.get('direction', '?')})"
        if action == "second_derivative":
            return "deriv2"
        if action == "fourier_transform":
            return "fft"
        if action == "inverse_fourier_transform":
            return "ifft"
        return action

    def _operation_color(self, action):
        """Return a BGR color for operation badges"""
        color_map = {
            "apply_blur": (245, 140, 66),
            "sharpen_image": (60, 190, 255),
            "flip_image": (120, 220, 120),
            "shift_image": (120, 170, 255),
            "change_brightness": (80, 220, 220),
            "change_contrast": (200, 200, 80),
            "split_channel": (160, 120, 255),
            "convert_grayscale": (180, 180, 180),
            "detect_edges": (90, 90, 255),
            "invert_image": (130, 130, 130),
            "first_derivative": (255, 140, 60),
            "second_derivative": (60, 255, 140),
            "fourier_transform": (140, 60, 255),
            "inverse_fourier_transform": (180, 90, 230),
        }
        return color_map.get(action, (180, 180, 180))

    def _draw_operation_badges(self, frame, operations):
        """Draw colored badges for active operations"""
        if not operations:
            return frame

        x = 10
        y = frame.shape[0] - 16
        max_badges = 5
        shown_ops = operations[:max_badges]

        for operation in shown_ops:
            label = self._format_operation_summary(operation)
            action = operation.get("action", "unknown")
            color = self._operation_color(action)

            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            badge_w = text_width + 16
            badge_h = text_height + 10

            cv2.rectangle(frame, (x, y - badge_h), (x + badge_w, y), color, -1)
            cv2.putText(
                frame,
                label,
                (x + 8, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

            x += badge_w + 8

        if len(operations) > max_badges:
            extra = f"+{len(operations) - max_badges}"
            (text_width, text_height), _ = cv2.getTextSize(extra, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            badge_w = text_width + 16
            badge_h = text_height + 10
            cv2.rectangle(frame, (x, y - badge_h), (x + badge_w, y), (110, 110, 110), -1)
            cv2.putText(
                frame,
                extra,
                (x + 8, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

        return frame

    def _draw_hud(self, frame, lines):
        """Draw a translucent HUD panel on the frame"""
        if not lines:
            return frame

        overlay = frame.copy()
        x = 10
        y = 10
        line_height = 24
        panel_width = 0
        for line in lines:
            (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            panel_width = max(panel_width, text_width)

        panel_width += 24
        panel_height = line_height * len(lines) + 16
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        text_y = y + 22
        for line in lines:
            cv2.putText(
                frame,
                line,
                (x + 12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            text_y += line_height

        return frame
    
    def _display_frame(self):
        """Display current frame with OpenCV"""
        try:
            with self.frame_lock:
                if self.current_frame is None:
                    return
                frame_to_display = self.current_frame.copy()
                active_operations = list(self.active_operations)
            
            for operation in active_operations:
                frame_to_display = self._apply_operation(frame_to_display, operation)
                if frame_to_display is None:
                    return

            if frame_to_display.ndim == 2:
                frame_to_display = cv2.cvtColor(frame_to_display, cv2.COLOR_GRAY2BGR)

            frame_to_display = np.ascontiguousarray(frame_to_display)

            now = time.monotonic()
            if self.last_display_time is not None:
                delta = now - self.last_display_time
                if delta > 0:
                    instant_fps = 1.0 / delta
                    self.display_fps = (
                        instant_fps if self.display_fps == 0.0 else (self.display_fps * 0.85 + instant_fps * 0.15)
                    )
            self.last_display_time = now

            active_labels = [self._format_operation_summary(op) for op in active_operations]
            if len(active_labels) > 4:
                active_labels = active_labels[:4] + [f"+{len(active_labels) - 4} more"]

            hud_lines = [
                f"Mode: {self.mode_name}",
                f"Resolution: {self.width}x{self.height} | Frames: {self.frame_count}",
                f"Target FPS: {self.fps} | Live FPS: {self.display_fps:.1f}",
                f"Active: {', '.join(active_labels) if active_labels else 'none'}",
                f"Last: {self.last_operation or 'none'}",
                "ENTER: listen | q: quit",
            ]
            
            self._draw_hud(frame_to_display, hud_lines)
            self._draw_operation_badges(frame_to_display, active_operations)
            
            cv2.imshow("Webcam Stream - Voice Agent", frame_to_display)
            cv2.waitKey(self.frame_delay)
        
        except Exception as e:
            print(f"[VIDEO ERROR] Display failed: {e}")
    
    def apply_operation(self, action, params):
        """Queue an operation for processing"""
        op = {"action": action, **params}
        try:
            self.operation_queue.put(op, block=False)
            print(f"[VIDEO] Operation queued: {action}")
        except queue.Full:
            print(f"[VIDEO WARNING] Operation queue full, skipping: {action}")
    
    def undo(self) -> bool:
        """Undo last operation by removing it from the active operation stack"""
        with self.frame_lock:
            if len(self.active_operations) > 0:
                removed_operation = self.active_operations.pop()
                self.frame_history.pop()
                if removed_operation.get("action") in {"fourier_transform", "inverse_fourier_transform"}:
                    self.last_fourier_spectrum = None
                self.last_operation = None
                print(f"[VIDEO] Undo: removed {removed_operation.get('action')} from active pipeline")
                return True
            else:
                print("[VIDEO] Nothing to undo")
                return False
    
    def reset(self):
        """Reset to original frame by clearing all active operations"""
        with self.frame_lock:
            if self.original_frame is not None:
                self.frame_history.clear()
                self.active_operations.clear()
                self.last_fourier_spectrum = None
                self.last_operation = None
                print("[VIDEO] Reset: cleared active operations")
            else:
                print("[VIDEO ERROR] Original frame not available")
    
    def stop(self):
        """Stop all threads and cleanup"""
        print("[VIDEO] Stopping capture...")
        self.is_running = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        print("[VIDEO] Stopped and cleaned up")
    
    def get_statistics(self):
        """Get processing statistics"""
        return {
            "frames_captured": self.frame_count,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "history_size": len(self.frame_history),
            "last_operation": self.last_operation
        }
