import json
import requests
import re

class CommandAgent:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "llama3.1:8b"
        self.request_timeout = 45  # Increased from 30 to handle busy Ollama
        
        # --- AJANIN HAFIZASI ---
        self.chat_history = [] 
        self.pending_command = None

        self.system_prompt = """
You are a STRICT intent-to-JSON parser for an image editing voice assistant.
Return ONLY valid JSON, no other text.
Return exactly one JSON object and nothing else.

CRITICAL RULE: BE CONSERVATIVE. If you are not 100% sure about a parameter, set it to null.
NEVER hallucinate or guess parameters. Do not invent information.
If the user's speech is unclear or does not explicitly mention an action, return {"action":"unknown","message":"..."}.

IMPORTANT: Use raw operation-style parameters aligned with the backend.

RAW PARAMETER TARGETS (MUST MATCH):
- shift direction: up | down | left | right
- flip axis: h | v
- channel color: r | g | b

ALLOWED ACTIONS (ONLY THESE):
1) {"action":"shift_image","direction":"up|down|left|right|null","pixels":integer|null}
2) {"action":"flip_image","axis":"h|v|null"}
3) {"action":"invert_image"}
4) {"action":"change_contrast","percentage":integer|null}
5) {"action":"change_brightness","percentage":integer|null}
6) {"action":"apply_blur","intensity":integer|null}
7) {"action":"sharpen_image","intensity":integer|null}
8) {"action":"convert_grayscale"}
9) {"action":"detect_edges"}
10) {"action":"split_channel","color":"r|g|b|null"}
11) {"action":"first_derivative","direction":"x|y|both|null"}
12) {"action":"second_derivative"}
13) {"action":"fourier_transform"}
14) {"action":"inverse_fourier_transform"}
15) {"action":"undo"}
16) {"action":"reset"}

REQUIRED-PARAMETER RULES:
- shift_image needs direction and pixels.
- flip_image needs axis.
- change_contrast and change_brightness need percentage.
- apply_blur and sharpen_image need intensity.
- split_channel needs color (red, green, or blue EXPLICITLY mentioned).
- first_derivative needs direction (x, y, or both).
- If a required parameter is missing or NOT explicitly mentioned, set it to null.
- Never invent default values. Do NOT guess colors.
- Convert spoken numbers to integers when possible ("twenty five" -> 25).
- Interpret vague intensities: "a lot" -> 40, "slightly" -> 3, "normal" -> 12, "max" -> 99.

MEMORY RULE:
- If previous context has an unfinished command with a null numeric field and the user now says only a number, complete and return the full command JSON.

MAPPING HINTS (ONLY if explicitly mentioned):
- "north", "up", "upwards" -> direction:"up" (ONLY if user says directional word)
- "south", "down", "downwards" -> direction:"down"
- "west", "left", "to the left" -> direction:"left"
- "east", "right", "to the right" -> direction:"right"
- "horizontal flip", "mirror left right" -> flip_image + axis:"h"
- "vertical flip", "mirror up down" -> flip_image + axis:"v"
- "red channel" -> split_channel + color:"r" (ONLY if "red" is EXPLICITLY mentioned)
- "green channel" -> split_channel + color:"g" (ONLY if "green" is EXPLICITLY mentioned)
- "blue channel" -> split_channel + color:"b" (ONLY if "blue" is EXPLICITLY mentioned)
- "invert", "negative", "reverse colors" -> invert_image
- "black and white" -> convert_grayscale
- "edge detection", "find edges", "outline" -> detect_edges
- "first derivative", "first order", "horizontal gradient", "vertical gradient" -> first_derivative + direction
- "second derivative", "second order", "laplacian" -> second_derivative
- "fourier", "frequency", "spectrum", "fft" -> fourier_transform
- "inverse fourier", "inverse fft", "ifft", "reconstruct image" -> inverse_fourier_transform
- "undo", "go back", "revert", "undo that", "take back" -> undo
- "reset", "start over", "begin again", "restore original" -> reset

UNSUPPORTED OR UNCLEAR REQUESTS:
- If intent is not one of the supported actions, return:
  {"action":"unknown","message":"Use one of the supported image commands."}
- If user's words don't CLEARLY indicate a specific color/direction/axis/intensity, return null for that param.
- Examples of ambiguous requests that should become unknown:
  - "let's channels" -> {"action":"unknown","message":"Which color: red, green, or blue?"}
  - "do something" -> {"action":"unknown","message":"Please specify an image operation."}
  - "make it better" -> {"action":"unknown","message":"Try: blur, sharpen, flip, invert, etc."}

EXAMPLES (CONSERVATIVE APPROACH):
User: "flip vertically"
Output: {"action":"flip_image","axis":"v"}

User: "shift image north"
Output: {"action":"shift_image","direction":"up","pixels":null}

User: "200"
Output: {"action":"shift_image","direction":"up","pixels":200}

User: "Please let's channels"
Output: {"action":"unknown","message":"Please specify which channel: red, green, or blue."}

User: "change something"
Output: {"action":"unknown","message":"Use one of the supported image commands."}

User: "blur"
Output: {"action":"apply_blur","intensity":null}

User: "undo"
Output: {"action":"undo"}

User: "go back"
Output: {"action":"undo"}

User: "reset"
Output: {"action":"reset"}

User: "start over"
Output: {"action":"reset"}
"""

    def _extract_json(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Safe bracket-balanced JSON extraction
            brace_count = 0
            start = -1
            for i, c in enumerate(text):
                if c == '{':
                    if brace_count == 0:
                        start = i
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError:
                            pass
        return {"action": "error", "message": "Failed to parse JSON."}

    def _get_intensity_word_mapping(self):
        """Maps vague intensity words to reasonable defaults."""
        return {
            "a little": 3, "slightly": 3, "tiny": 2,
            "a bit": 5, "small": 5,
            "normal": 12, "medium": 15,
            "much": 30, "a lot": 40, "very much": 50,
            "extreme": 70, "very extreme": 80, "max": 99,
        }

    def _parse_number(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)

        raw = str(value).strip().lower()
        if raw.isdigit():
            return int(raw)

        # Check intensity word mappings first
        intensity_words = self._get_intensity_word_mapping()
        if raw in intensity_words:
            return intensity_words[raw]

        word_map = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
            "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
            "hundred": 100
        }

        parts = re.split(r"[\s-]+", raw)
        total = 0
        current = 0

        for part in parts:
            if part not in word_map:
                return None
            number = word_map[part]
            if number == 100:
                current = max(1, current) * 100
            else:
                current += number

        total += current
        return total if total > 0 or raw == "zero" else None

    def _number_from_user_text(self, user_text):
        text = user_text.strip().lower()
        if text.isdigit():
            return int(text)
        return self._parse_number(text)

    def _direction_from_user_text(self, user_text):
        text = user_text.strip().lower()
        if re.search(r"\b(both|all directions)\b", text):
            return "both"
        if re.search(r"\b(horizontal|x)\b", text):
            return "x"
        if re.search(r"\b(vertical|y)\b", text):
            return "y"
        return None

    def _validate_direction(self, direction):
        if direction is None:
            return None

        normalized = str(direction).strip().lower()
        alias_map = {
            "n": "up",
            "north": "up",
            "up": "up",
            "upwards": "up",
            "towards the top": "up",
            "s": "down",
            "south": "down",
            "down": "down",
            "downwards": "down",
            "towards the bottom": "down",
            "w": "left",
            "west": "left",
            "left": "left",
            "to the left": "left",
            "e": "right",
            "east": "right",
            "right": "right",
            "to the right": "right",
        }
        result = alias_map.get(normalized)
        if result is None and normalized not in ["", "null"]:
            # LLM hallucinated an invalid direction
            print(f"[WARN] LLM hallucinated invalid direction: '{direction}' - resetting to null")
        return result

    def _validate_color(self, color):
        if color is None:
            return None

        normalized = str(color).strip().lower()
        alias_map = {
            "r": "red",
            "red": "red",
            "g": "green",
            "green": "green",
            "b": "blue",
            "blue": "blue",
        }
        result = alias_map.get(normalized)
        if result is None and normalized not in ["", "null"]:
            # LLM hallucinated an invalid color - log it
            print(f"[WARN] LLM hallucinated invalid color: '{color}' - resetting to null")
        return result

    def _validate_axis(self, axis):
        if axis is None:
            return None

        normalized = str(axis).strip().lower()
        alias_map = {
            "h": "horizontal",
            "horizontal": "horizontal",
            "v": "vertical",
            "vertical": "vertical",
        }
        return alias_map.get(normalized)

    def _normalize_action(self, data):
        if not isinstance(data, dict):
            return {"action": "error", "message": "Invalid response format."}

        action = data.get("action")

        if action == "shift_image":
            direction = self._validate_direction(data.get("direction"))
            pixels = self._parse_number(data.get("pixels"))
            return {"action": "shift_image", "direction": direction, "pixels": pixels}

        if action == "flip_image":
            axis = self._validate_axis(data.get("axis"))
            return {"action": "flip_image", "axis": axis}

        if action == "invert_image":
            return {"action": "invert_image"}

        if action in {"change_contrast", "change_brightness"}:
            percentage = self._parse_number(data.get("percentage"))
            return {"action": action, "percentage": percentage}

        if action in {"apply_blur", "sharpen_image"}:
            intensity = self._parse_number(data.get("intensity"))
            return {"action": action, "intensity": intensity}

        if action == "split_channel":
            color = self._validate_color(data.get("color"))
            return {"action": "split_channel", "color": color}

        if action == "first_derivative":
            direction = data.get("direction")
            # Normalize direction: x, y, both
            if direction is not None:
                direction = str(direction).strip().lower()
                if direction not in ["x", "y", "both"]:
                    direction = None
            return {"action": "first_derivative", "direction": direction}

        if action == "second_derivative":
            return {"action": "second_derivative"}

        if action == "fourier_transform":
            return {"action": "fourier_transform"}

        if action == "inverse_fourier_transform":
            return {"action": "inverse_fourier_transform"}

        if action in {"convert_grayscale", "detect_edges", "undo", "reset"}:
            return {"action": action}

        message = data.get("message") if isinstance(data.get("message"), str) else "Use a valid image command."
        return {"action": "unknown", "message": message}

    def _merge_with_pending_command(self, user_text):
        if not self.pending_command:
            return None

        spoken_number = self._number_from_user_text(user_text)
        spoken_direction = self._direction_from_user_text(user_text)
        if spoken_number is None:
            merged = dict(self.pending_command)
            action = merged.get("action")
            if action == "first_derivative" and merged.get("direction") is None and spoken_direction is not None:
                merged["direction"] = spoken_direction
                return merged
            return None

        merged = dict(self.pending_command)
        action = merged.get("action")

        if action == "shift_image" and merged.get("pixels") is None:
            merged["pixels"] = spoken_number
            return merged
        if action in {"change_contrast", "change_brightness"} and merged.get("percentage") is None:
            merged["percentage"] = spoken_number
            return merged
        if action in {"apply_blur", "sharpen_image"} and merged.get("intensity") is None:
            merged["intensity"] = spoken_number
            return merged

        return None

    def _store_pending_if_needed(self, command_data):
        action = command_data.get("action")
        needs_more = (
            (action == "shift_image" and (command_data.get("direction") is None or command_data.get("pixels") is None))
            or (action == "flip_image" and command_data.get("axis") is None)
            or (action in {"change_contrast", "change_brightness"} and command_data.get("percentage") is None)
            or (action in {"apply_blur", "sharpen_image"} and command_data.get("intensity") is None)
            or (action == "split_channel" and command_data.get("color") is None)
            or (action == "first_derivative" and command_data.get("direction") is None)
        )
        self.pending_command = command_data if needs_more else None
    
    def clear_history(self):
        """İşlem başarıyla tamamlandığında hafızayı temizler (eski komutların karışmasını önler)."""
        self.chat_history = []
        self.pending_command = None
        print("[AGENT] Memory cleared for the next command.")

    def process_command(self, user_text):
        print(f"\n[AGENT] Analyzing: '{user_text}'")

        merged_pending = self._merge_with_pending_command(user_text)
        if merged_pending is not None:
            print("[AGENT] Merged numeric follow-up with pending command.")
            self._store_pending_if_needed(merged_pending)
            return merged_pending
        
        history_text = "\n".join(self.chat_history[-8:])
        full_prompt = f"{self.system_prompt}\n\nRecent Conversation:\n{history_text}\n\nCurrent User Speech: '{user_text}'\nJSON Output:"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json"
        }

        try:
            print("[AGENT] Thinking...")
            response = requests.post(self.ollama_url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
            
            result_text = response.json().get("response", "")
            print(f"[AGENT] Raw Output: {result_text}")
            
            command_data = self._normalize_action(self._extract_json(result_text))
            self._store_pending_if_needed(command_data)

            self.chat_history.append(f"User: {user_text}")
            self.chat_history.append(f"Agent JSON: {json.dumps(command_data, ensure_ascii=True)}")

            return command_data

        except Exception as e:
            print(f"[ERROR] Agent communication failed: {e}")
            return {"action": "error", "message": "I'm having trouble thinking right now."}