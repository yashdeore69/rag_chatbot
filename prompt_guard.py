from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import re

class PromptGuard:
    def __init__(self, model_name="meta-llama/Llama-Prompt-Guard-2-86M"):
        """
        Initialize Meta's Prompt Guard 2 model.
        
        Note: Prompt Guard 2 uses binary classification (BENIGN/MALICIOUS)
        instead of the three-class system (BENIGN/INJECTION/JAILBREAK) from v1.
        """
        print("   Loading Prompt Guard 2 model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            print("   Prompt Guard 2 loaded successfully!")
            
            # Prompt Guard 2 uses binary classification
            # id2label: {0: "BENIGN", 1: "MALICIOUS"}
            self.labels = self.model.config.id2label
            print(f"   Model labels: {self.labels}")
            
        except Exception as e:
            print(f"   Warning: Could not load Prompt Guard 2: {e}")
            print("   Continuing without Prompt Guard...")
            self.model = None
            self.tokenizer = None
            self.labels = None
    
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
    
    def check_prompt(self, text: str, threshold: float = 0.5) -> dict:
        """
        Multi-layered prompt injection detection using Prompt Guard 2.
        
        Strategy:
        1. First check for obvious manipulation patterns (fast, no false positives)
        2. If suspicious, use Prompt Guard 2 ML model (binary classification)
        3. Only block if BOTH agree it's dangerous OR ML has very high confidence
        
        Args:
            text: The input text to check
            threshold: Confidence threshold for malicious classification (0.0-1.0)
                      Default 0.5 provides balanced detection
        """
        
        # Layer 1: Pattern-based detection (catches obvious attacks)
        has_suspicious_pattern = self.has_instruction_patterns(text)
        
        # If model not loaded, rely on pattern matching only
        if self.model is None or self.tokenizer is None:
            if has_suspicious_pattern:
                return {
                    "is_safe": False,
                    "label": "MALICIOUS",
                    "score": 1.0,
                    "detection_method": "pattern_matching",
                    "probabilities": {"benign": 0.0, "malicious": 1.0},
                    "message": "Suspicious instruction pattern detected"
                }
            return {
                "is_safe": True,
                "label": "BENIGN",
                "score": 0.0,
                "detection_method": "pattern_matching",
                "probabilities": {"benign": 1.0, "malicious": 0.0},
                "message": "Query appears safe"
            }
        
        try:
            # Layer 2: ML-based detection with Prompt Guard 2
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
            
            # Prompt Guard 2 uses binary classification
            # Index 0: BENIGN, Index 1: MALICIOUS
            benign_score = probabilities[0].item()
            malicious_score = probabilities[1].item()
            
            # Determine predicted class
            predicted_class = torch.argmax(probabilities).item()
            predicted_label = self.labels[predicted_class]
            
            # Multi-layered decision logic:
            # Block if BOTH pattern matching AND ML model agree, OR if ML has very high confidence
            ml_flags_as_unsafe = (malicious_score > threshold)
            
            if has_suspicious_pattern and ml_flags_as_unsafe:
                # Both agree - high confidence block
                return {
                    "is_safe": False,
                    "label": predicted_label,
                    "score": malicious_score,
                    "detection_method": "pattern_and_ml",
                    "probabilities": {
                        "benign": benign_score,
                        "malicious": malicious_score
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
                        "malicious": malicious_score
                    },
                    "message": "Suspicious pattern detected but ML model says benign - allowing"
                }
            elif ml_flags_as_unsafe and malicious_score > 0.85:
                # ML very confident even without pattern - block
                # Using 0.85 threshold for high-confidence blocking without pattern match
                return {
                    "is_safe": False,
                    "label": predicted_label,
                    "score": malicious_score,
                    "detection_method": "ml_high_confidence",
                    "probabilities": {
                        "benign": benign_score,
                        "malicious": malicious_score
                    },
                    "message": "Very high ML confidence of malicious prompt"
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
                        "malicious": malicious_score
                    },
                    "message": "Query appears safe"
                }
            
        except Exception as e:
            print(f"   Error in Prompt Guard 2 check: {e}")
            # On error, fall back to pattern matching
            if has_suspicious_pattern:
                return {
                    "is_safe": False,
                    "label": "MALICIOUS",
                    "score": 1.0,
                    "detection_method": "pattern_fallback",
                    "probabilities": {"benign": 0.0, "malicious": 1.0},
                    "message": "Error in ML check, blocked by pattern matching"
                }
            return {
                "is_safe": True,
                "label": "ERROR",
                "score": 0.0,
                "detection_method": "error",
                "probabilities": {"benign": 0.0, "malicious": 0.0},
                "message": "Error during check but no suspicious patterns found"
            }

# Singleton instance
_prompt_guard_instance = None

def get_prompt_guard():
    global _prompt_guard_instance
    if _prompt_guard_instance is None:
        _prompt_guard_instance = PromptGuard()
    return _prompt_guard_instance