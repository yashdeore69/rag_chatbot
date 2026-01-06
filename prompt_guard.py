from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import re

class PromptGuard:
    def __init__(self, model_name="meta-llama/Prompt-Guard-86M"):
        """
        Initialize Meta's Prompt Guard model.
        """
        print("   Loading Prompt Guard model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            print("   Prompt Guard loaded successfully!")
        except Exception as e:
            print(f"   Warning: Could not load Prompt Guard: {e}")
            print("   Continuing without Prompt Guard...")
            self.model = None
            self.tokenizer = None
    
    def has_instruction_patterns(self, text: str) -> bool:
        """
        Check for explicit instruction/manipulation patterns.
        This catches obvious prompt injection attempts.
        """
        text_lower = text.lower()
        
        # Patterns that indicate prompt manipulation
        suspicious_patterns = [
            r'ignore\s+(previous|above|all|your)',
            r'disregard\s+(previous|above|all|your)',
            r'forget\s+(previous|above|all|your)',
            r'system\s+prompt',
            r'you\s+are\s+now',
            r'act\s+as\s+(?!a\s+(?:teacher|tutor|guide))',  # Allow "act as a teacher"
            r'pretend\s+(?:you|to\s+be)',
            r'roleplay\s+as',
            r'new\s+instructions?',
            r'override\s+',
            r'<\|.*?\|>',  # Special tokens
            r'###\s*System',
            r'###\s*Assistant',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def check_prompt(self, text: str, threshold: float = 0.85) -> dict:
        """
        Multi-layered prompt injection detection.
        
        Strategy:
        1. First check for obvious manipulation patterns (fast, no false positives)
        2. If suspicious, use Prompt Guard ML model (slower, some false positives)
        3. Only block if BOTH agree it's dangerous
        """
        
        # Layer 1: Pattern-based detection (catches obvious attacks)
        has_suspicious_pattern = self.has_instruction_patterns(text)
        
        # If model not loaded, rely on pattern matching only
        if self.model is None or self.tokenizer is None:
            if has_suspicious_pattern:
                return {
                    "is_safe": False,
                    "label": "INJECTION",
                    "score": 1.0,
                    "detection_method": "pattern_matching",
                    "probabilities": {"benign": 0.0, "injection": 1.0, "jailbreak": 0.0},
                    "message": "Suspicious instruction pattern detected"
                }
            return {
                "is_safe": True,
                "label": "BENIGN",
                "score": 0.0,
                "detection_method": "pattern_matching",
                "probabilities": {"benign": 1.0, "injection": 0.0, "jailbreak": 0.0},
                "message": "Query appears safe"
            }
        
        try:
            # Layer 2: ML-based detection
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            probabilities = softmax(logits, dim=-1)[0]
            
            # Get scores for each category
            benign_score = probabilities[0].item()
            injection_score = probabilities[1].item()
            jailbreak_score = probabilities[2].item()
            
            # Determine predicted class
            predicted_class = torch.argmax(probabilities).item()
            labels = ["BENIGN", "INJECTION", "JAILBREAK"]
            predicted_label = labels[predicted_class]
            
            # Multi-layered decision logic:
            # Block only if BOTH pattern matching AND ML model agree
            ml_flags_as_unsafe = (jailbreak_score > threshold)
            
            if has_suspicious_pattern and ml_flags_as_unsafe:
                # Both agree - high confidence block
                return {
                    "is_safe": False,
                    "label": predicted_label,
                    "score": jailbreak_score,
                    "detection_method": "pattern_and_ml",
                    "probabilities": {
                        "benign": benign_score,
                        "injection": injection_score,
                        "jailbreak": jailbreak_score
                    },
                    "message": "High-confidence prompt injection detected"
                }
            elif has_suspicious_pattern:
                # Pattern match but ML disagrees - treat as suspicious but allow with warning
                return {
                    "is_safe": True,
                    "label": "SUSPICIOUS",
                    "score": 0.5,
                    "detection_method": "pattern_only",
                    "probabilities": {
                        "benign": benign_score,
                        "injection": injection_score,
                        "jailbreak": jailbreak_score
                    },
                    "message": "Suspicious pattern detected but ML model says benign - allowing"
                }
            elif ml_flags_as_unsafe and jailbreak_score > 0.95:
                # ML very confident even without pattern - block
                return {
                    "is_safe": False,
                    "label": predicted_label,
                    "score": jailbreak_score,
                    "detection_method": "ml_high_confidence",
                    "probabilities": {
                        "benign": benign_score,
                        "injection": injection_score,
                        "jailbreak": jailbreak_score
                    },
                    "message": "Very high ML confidence of jailbreak"
                }
            else:
                # Safe
                return {
                    "is_safe": True,
                    "label": "BENIGN",
                    "score": benign_score,
                    "detection_method": "ml",
                    "probabilities": {
                        "benign": benign_score,
                        "injection": injection_score,
                        "jailbreak": jailbreak_score
                    },
                    "message": "Query appears safe"
                }
            
        except Exception as e:
            print(f"   Error in Prompt Guard check: {e}")
            # On error, fall back to pattern matching
            if has_suspicious_pattern:
                return {
                    "is_safe": False,
                    "label": "INJECTION",
                    "score": 1.0,
                    "detection_method": "pattern_fallback",
                    "probabilities": {"benign": 0.0, "injection": 1.0, "jailbreak": 0.0},
                    "message": "Error in ML check, blocked by pattern matching"
                }
            return {
                "is_safe": True,
                "label": "ERROR",
                "score": 0.0,
                "detection_method": "error",
                "probabilities": {"benign": 0.0, "injection": 0.0, "jailbreak": 0.0},
                "message": "Error during check but no suspicious patterns found"
            }

# Singleton instance
_prompt_guard_instance = None

def get_prompt_guard():
    global _prompt_guard_instance
    if _prompt_guard_instance is None:
        _prompt_guard_instance = PromptGuard()
    return _prompt_guard_instance


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# class PromptGuard:
#     def __init__(self, model_name="meta-llama/Prompt-Guard-86M"):
#         """
#         Initialize Meta's Prompt Guard model.
#         Model will be downloaded on first use and cached locally.
#         """
#         print("   Loading Prompt Guard model (this may take a moment on first run)...")
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
#             self.model.eval()  # Set to evaluation mode
#             print("   Prompt Guard model loaded successfully!")
#         except Exception as e:
#             print(f"   Warning: Could not load Prompt Guard model: {e}")
#             print("   Continuing without Prompt Guard protection...")
#             self.model = None
#             self.tokenizer = None
        
#     def check_prompt(self, text: str, threshold: float = 0.75) -> dict:
#         """
#         Check if a prompt contains injection or jailbreak attempts.
        
#         Args:
#             text: The user's query text
#             threshold: Confidence threshold (0-1) for flagging prompts
            
#         Returns:
#             dict with keys:
#                 - is_safe: bool indicating if prompt is safe
#                 - label: classification label (BENIGN, INJECTION, or JAILBREAK)
#                 - score: confidence score
#                 - probabilities: dict of all class probabilities
#                 - message: human-readable message
#         """
#         # If model failed to load, default to safe
#         if self.model is None or self.tokenizer is None:
#             return {
#                 "is_safe": True,
#                 "label": "BENIGN",
#                 "score": 1.0,
#                 "probabilities": {"benign": 1.0, "injection": 0.0, "jailbreak": 0.0},
#                 "message": "Prompt Guard not available - defaulting to safe"
#             }
        
#         try:
#             # Tokenize input
#             inputs = self.tokenizer(
#                 text, 
#                 return_tensors="pt", 
#                 truncation=True, 
#                 max_length=512,
#                 padding=True
#             )
            
#             # Get model predictions
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 logits = outputs.logits
#                 probabilities = torch.softmax(logits, dim=-1)[0]
            
#             # Get predicted class
#             predicted_class = torch.argmax(probabilities).item()
#             confidence = probabilities[predicted_class].item()
            
#             # Map class indices to labels
#             # 0: BENIGN, 1: INJECTION, 2: JAILBREAK
#             labels = ["BENIGN", "INJECTION", "JAILBREAK"]
#             predicted_label = labels[predicted_class]
            
#             # Determine if prompt is safe
#             is_safe = predicted_label == "BENIGN" and confidence > threshold
            
#             # Create response
#             result = {
#                 "is_safe": is_safe,
#                 "label": predicted_label,
#                 "score": confidence,
#                 "probabilities": {
#                     "benign": probabilities[0].item(),
#                     "injection": probabilities[1].item(),
#                     "jailbreak": probabilities[2].item()
#                 }
#             }
            
#             # Add human-readable message
#             if is_safe:
#                 result["message"] = "Query appears safe."
#             elif predicted_label == "INJECTION":
#                 result["message"] = "Potential prompt injection detected."
#             else:
#                 result["message"] = "Potential jailbreak attempt detected."
            
#             return result
            
#         except Exception as e:
#             print(f"   Error in Prompt Guard check: {e}")
#             # Fail safe - flag as potentially unsafe on error
#             return {
#                 "is_safe": False,
#                 "label": "ERROR",
#                 "score": 0.0,
#                 "probabilities": {"benign": 0.0, "injection": 0.0, "jailbreak": 0.0},
#                 "message": f"Error during safety check: {str(e)}"
#             }

# # Create a singleton instance
# _prompt_guard_instance = None

# def get_prompt_guard():
#     """
#     Get or create the PromptGuard instance (singleton pattern).
#     This ensures the model is only loaded once.
#     """
#     global _prompt_guard_instance
#     if _prompt_guard_instance is None:
#         _prompt_guard_instance = PromptGuard()
#     return _prompt_guard_instance


