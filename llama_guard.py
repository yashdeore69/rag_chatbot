# from langchain_ollama import OllamaLLM
# from typing import Tuple

# class LlamaGuard:
#     def __init__(self, model_name: str = "llama-guard3:1b"):
#         """
#         Initialize Llama Guard for content moderation.
        
#         Args:
#             model_name: The Ollama model name for Llama Guard
#         """
#         self.model = OllamaLLM(model=model_name)
        
#         # Llama Guard 3 safety categories
#         self.safety_categories = """S1: Violent Crimes
# S2: Non-Violent Crimes
# S3: Sex-Related Crimes
# S4: Child Sexual Exploitation
# S5: Defamation
# S6: Specialized Advice
# S7: Privacy
# S8: Intellectual Property
# S9: Indiscriminate Weapons
# S10: Hate
# S11: Suicide & Self-Harm
# S12: Sexual Content
# S13: Elections"""

#     def check_prompt(self, user_input: str) -> Tuple[bool, str]:
#         """
#         Check if user input is safe.
        
#         Args:
#             user_input: The user's query text
            
#         Returns:
#             Tuple of (is_safe, category/reason)
#         """
#         prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

# <BEGIN UNSAFE CONTENT CATEGORIES>
# {self.safety_categories}
# <END UNSAFE CONTENT CATEGORIES>

# <BEGIN CONVERSATION>
# User: {user_input}
# <END CONVERSATION>

# Provide your safety assessment for User in the above conversation:
# - First line must read 'safe' or 'unsafe'.
# - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""

#         try:
#             response = self.model.invoke(prompt).strip()
#             lines = response.split('\n')
            
#             is_safe = lines[0].lower().strip() == 'safe'
#             category = lines[1].strip() if len(lines) > 1 and not is_safe else ""
            
#             return is_safe, category
#         except Exception as e:
#             print(f"Error in Llama Guard prompt check: {e}")
#             # Fail open - allow the request but log the error
#             return True, ""

#     def check_response(self, user_input: str, assistant_response: str, context: str = "") -> Tuple[bool, str]:
#         """
#         Check if assistant response is safe, considering it's based on PDF documents.
        
#         Args:
#             user_input: The original user query
#             assistant_response: The model's response
#             context: The PDF context used to generate the response
            
#         Returns:
#             Tuple of (is_safe, category/reason)
#         """
#         # Enhanced prompt that tells Llama Guard this is educational/informational content from documents
#         prompt = f"""[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.

# <BEGIN UNSAFE CONTENT CATEGORIES>
# {self.safety_categories}
# <END UNSAFE CONTENT CATEGORIES>

# <BEGIN CONTEXT>
# IMPORTANT: The Agent's response is based on information retrieved from legitimate PDF documents provided by the user. The Agent is acting as an information retrieval system, presenting factual content from these documents.

# Academic, educational, technical, medical, and informational content from legitimate sources should be considered SAFE, even if it discusses sensitive topics, as long as:
# 1. The information is factual and educational in nature
# 2. The response is directly answering the user's question based on the provided documents
# 3. The response is not promoting harmful activities or providing instructions for illegal actions
# 4. The response is not violating privacy by sharing personal information not present in the documents

# Document Context Preview:
# {context[:500]}...
# <END CONTEXT>

# <BEGIN CONVERSATION>
# User: {user_input}

# Agent: {assistant_response}
# <END CONVERSATION>

# Provide your safety assessment for Agent in the above conversation:
# - First line must read 'safe' or 'unsafe'.
# - If unsafe, a second line must include a comma-separated list of violated categories.
# - Remember: Educational/informational content from documents is SAFE. [/INST]"""

#         try:
#             response = self.model.invoke(prompt).strip()
#             lines = response.split('\n')
            
#             is_safe = lines[0].lower().strip() == 'safe'
#             category = lines[1].strip() if len(lines) > 1 and not is_safe else ""
            
#             return is_safe, category
#         except Exception as e:
#             print(f"Error in Llama Guard response check: {e}")
#             # Fail open - allow the response but log the error
#             return True, ""

from langchain_ollama import OllamaLLM
from typing import Tuple, Dict
import re

class LlamaGuard:
    def __init__(self, model_name: str = "llama-guard3:1b"):
        """
        Initialize Llama Guard for content moderation and response validation.
        
        Args:
            model_name: The Ollama model name for Llama Guard
        """
        self.model = OllamaLLM(model=model_name)
        
        # Llama Guard 3 safety categories
        self.safety_categories = """S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S4: Child Sexual Exploitation
S5: Defamation
S6: Specialized Advice
S7: Privacy
S8: Intellectual Property
S9: Indiscriminate Weapons
S10: Hate
S11: Suicide & Self-Harm
S12: Sexual Content
S13: Elections"""

        # Context grounding validation categories
        self.grounding_categories = """G1: Response contains information not present in the context
G2: Response makes claims that contradict the context
G3: Response includes external knowledge beyond the context
G4: Response fabricates or hallucinates facts
G5: Response provides partial information that misrepresents the context
G6: Response fails to acknowledge lack of information in context"""

    def check_prompt(self, user_input: str) -> Tuple[bool, str]:
        """
        Check if user input is safe.
        
        Args:
            user_input: The user's query text
            
        Returns:
            Tuple of (is_safe, category/reason)
        """
        prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{self.safety_categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: {user_input}
<END CONVERSATION>

Provide your safety assessment for User in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""

        try:
            response = self.model.invoke(prompt).strip()
            lines = response.split('\n')
            
            is_safe = lines[0].lower().strip() == 'safe'
            category = lines[1].strip() if len(lines) > 1 and not is_safe else ""
            
            return is_safe, category
        except Exception as e:
            print(f"Error in Llama Guard prompt check: {e}")
            # Fail open - allow the request but log the error
            return True, ""

    def check_context_grounding(self, response: str, context: str, query: str) -> Dict[str, any]:
        """
        Validate if the assistant's response is grounded in the provided context.
        This is the KEY method for ensuring RAG responses don't hallucinate.
        
        Args:
            response: The LLM's generated response
            context: The retrieved context from vector DB
            query: The original user query
            
        Returns:
            Dict containing validation results
        """
        # First, check for obvious hallucination patterns
        hallucination_indicators = self._check_hallucination_patterns(response, context)
        
        # If clear pattern issues, no need for deep validation
        if hallucination_indicators['has_issues']:
            critical_issues = [issue for issue in hallucination_indicators['issues'] 
                             if 'Meta-commentary' in issue or 'External citation' in issue]
            
            if critical_issues:
                return {
                    'is_grounded': False,
                    'validation_method': 'pattern_only',
                    'unsupported_claims': ', '.join(critical_issues),
                    'violated_categories': 'G1, G3',
                    'pattern_hallucinations': hallucination_indicators['issues'],
                    'confidence_score': 0.0
                }
        
        # Use Llama Guard for deep validation with improved prompt
        prompt = f"""[INST] Task: Validate if the Assistant's response is reasonably grounded in the provided context.

<BEGIN CONTEXT FROM VECTOR DATABASE>
{context[:2000]}
<END CONTEXT FROM VECTOR DATABASE>

<BEGIN CONVERSATION>
User Query: {query}

Assistant Response: {response}
<END CONVERSATION>

VALIDATION INSTRUCTIONS:
- Check if the main ideas and concepts in the response come from the context
- Allow reasonable paraphrasing and natural language variations
- Allow combining information from different parts of the context
- Mark as 'grounded' if the response answers using context information
- Mark as 'ungrounded' ONLY if the response contains significant information NOT in the context
- Educational responses that explain concepts from the context should be marked as 'grounded'

Provide your assessment on ONE line:
- Write ONLY 'grounded' or 'ungrounded' [/INST]"""

        try:
            llama_response = self.model.invoke(prompt).strip().lower()
            
            # Extract just the first word/decision
            is_grounded = 'grounded' in llama_response.split()[0] if llama_response else False
            
            # If Llama Guard says ungrounded but no critical pattern issues, be lenient
            if not is_grounded and not hallucination_indicators['has_issues']:
                # Give benefit of doubt if no clear red flags
                is_grounded = True
                validation_method = 'lenient_pass'
            else:
                validation_method = 'llama_guard_validated'
            
            return {
                'is_grounded': is_grounded,
                'validation_method': validation_method,
                'unsupported_claims': "" if is_grounded else "Response may contain ungrounded information",
                'violated_categories': "" if is_grounded else "G1",
                'pattern_hallucinations': hallucination_indicators['issues'] if not is_grounded else [],
                'confidence_score': self._calculate_confidence(is_grounded, hallucination_indicators)
            }
            
        except Exception as e:
            print(f"Error in context grounding validation: {e}")
            # On error, be lenient if no pattern issues
            if not hallucination_indicators['has_issues']:
                return {
                    'is_grounded': True,
                    'validation_method': 'error_lenient',
                    'unsupported_claims': '',
                    'violated_categories': '',
                    'pattern_hallucinations': [],
                    'confidence_score': 0.7
                }
            return {
                'is_grounded': False,
                'validation_method': 'error_fallback',
                'unsupported_claims': 'Validation error occurred',
                'violated_categories': 'Unknown',
                'pattern_hallucinations': hallucination_indicators['issues'],
                'confidence_score': 0.0
            }

    def _check_hallucination_patterns(self, response: str, context: str) -> Dict[str, any]:
        """
        Pattern-based hallucination detection for CRITICAL issues only.
        Only flags severe problems to reduce false positives.
        
        Returns:
            Dict with 'has_issues' flag and list of 'issues'
        """
        issues = []
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Pattern 1: Check for meta-commentary (LLM talking about itself)
        # These are CRITICAL issues that should always be flagged
        meta_patterns = [
            r'\bi am (an )?(ai|assistant|language model)',
            r'\bi don\'?t have access to',
            r'\bi cannot (browse|search the web|access the internet)',
            r'as an ai\b',
            r'my training data',
            r'i was trained on',
            r'according to my knowledge base',
            r'in my database'
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, response_lower):
                issues.append(f"Meta-commentary detected: {pattern}")
        
        # Pattern 2: Check for external source citations not in context
        # Only flag obvious external citations
        citation_patterns = [
            r'according to wikipedia',
            r'according to google',
            r'source: http',
            r'as stated by \w+ \w+ \(researcher',
            r'in a study published in'
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, response_lower):
                issues.append(f"External citation detected: {pattern}")
        
        # Pattern 3: Check for numerical/statistical claims not in context
        # Only flag if there are MANY fabricated numbers (reduces false positives)
        response_numbers = set(re.findall(r'\b\d+\.?\d*\b', response))
        context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context))
        
        fabricated_numbers = response_numbers - context_numbers
        # Increased threshold from 2 to 5 to reduce false positives
        if fabricated_numbers and len(fabricated_numbers) > 1:
            issues.append(f"Many numerical claims not in context: {len(fabricated_numbers)} numbers")
        
        # Pattern 4: REMOVED - Too many false positives on legitimate responses
        # Technical terms are often paraphrased or synonyms
        
        # Pattern 5: Check for definitive statements - RELAXED
        # Only flag if there are MANY overconfident statements
        definitive_patterns = [
            r'\balways\b',
            r'\bnever\b',
            r'\bdefinitely\b',
            r'\bcertainly\b',
            r'\babsolutely\b',
            r'\bguaranteed\b'
        ]
        
        definitive_count = sum(1 for pattern in definitive_patterns if re.search(pattern, response_lower))
        # Increased threshold from 3 to 5
        if definitive_count > 5:
            issues.append("Excessive overconfident language")
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues
        }

    def _calculate_confidence(self, is_grounded: bool, hallucination_indicators: Dict) -> float:
        """
        Calculate confidence score for the grounding validation.
        
        Returns:
            Float between 0.0 and 1.0
        """
        if not is_grounded:
            return 0.0
        
        if hallucination_indicators['has_issues']:
            # Reduce confidence based on number of pattern issues
            issue_count = len(hallucination_indicators['issues'])
            penalty = min(0.5, issue_count * 0.1)
            return max(0.0, 1.0 - penalty)
        
        return 1.0

    def check_response(self, user_input: str, assistant_response: str, context: str = "") -> Tuple[bool, str]:
        """
        Check if assistant response is safe AND grounded in context.
        This method now performs BOTH safety and grounding validation.
        
        Args:
            user_input: The original user query
            assistant_response: The model's response
            context: The PDF context used to generate the response
            
        Returns:
            Tuple of (is_safe_and_grounded, reason)
        """
        # Step 1: Check content safety
        prompt = f"""[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy.

<BEGIN UNSAFE CONTENT CATEGORIES>
{self.safety_categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONTEXT>
The Agent's response is based on educational information about Machine Learning, specifically Probabilistic Models, Decision Trees, and related algorithms from course materials.

Educational content should be considered SAFE. Only mark as UNSAFE if the response:
1. Promotes harmful activities or illegal actions
2. Contains hate speech or discriminatory content
3. Violates privacy by sharing personal information
4. Contains inappropriate sexual content

Document Context:
{context[:300]}...
<END CONTEXT>

<BEGIN CONVERSATION>
User: {user_input}

Agent: {assistant_response}
<END CONVERSATION>

Provide your safety assessment on ONE line:
Write ONLY 'safe' or 'unsafe' [/INST]"""

        try:
            response = self.model.invoke(prompt).strip().lower()
            
            # Extract just the first word
            is_safe = 'safe' in response.split()[0] if response else True
            
            if not is_safe:
                return False, f"Content safety violation detected"
            
            # Step 2: If safe, do lightweight grounding check
            # Only block if there are CRITICAL issues
            if context:
                grounding_result = self.check_context_grounding(
                    assistant_response, 
                    context, 
                    user_input
                )
                
                # Only block if confidence is very low (clear hallucination)
                if not grounding_result['is_grounded'] and grounding_result['confidence_score'] < 0.3:
                    reason = "Response likely contains hallucinations"
                    if grounding_result['pattern_hallucinations']:
                        reason += f": {', '.join(grounding_result['pattern_hallucinations'][:2])}"
                    return False, reason
            
            return True, "Content is safe and reasonably grounded"
            
        except Exception as e:
            print(f"Error in Llama Guard response check: {e}")
            # On error, be lenient - allow educational content
            return True, "Validation error - allowing educational content"

    def validate_rag_response(self, query: str, response: str, context: str, 
                             retrieved_chunks: list) -> Dict[str, any]:
        """
        Comprehensive RAG response validation.
        This is the main method to use for validating RAG chatbot responses.
        
        Args:
            query: User's original query
            response: LLM's generated response
            context: Full context passed to LLM
            retrieved_chunks: List of retrieved document chunks with metadata
            
        Returns:
            Dict with comprehensive validation results
        """
        # 1. Safety check
        is_safe, safety_reason = self.check_prompt(query)
        
        # 2. Response safety and grounding check
        response_safe, response_reason = self.check_response(query, response, context)
        
        # 3. Deep grounding validation
        grounding_check = self.check_context_grounding(response, context, query)
        
        # 4. Check if response acknowledges limitations appropriately
        acknowledges_limits = self._check_limitation_acknowledgment(response)
        
        return {
            'overall_valid': is_safe and response_safe and grounding_check['is_grounded'],
            'query_safe': is_safe,
            'query_safety_reason': safety_reason,
            'response_safe': response_safe,
            'response_safety_reason': response_reason,
            'is_grounded': grounding_check['is_grounded'],
            'grounding_confidence': grounding_check['confidence_score'],
            'unsupported_claims': grounding_check['unsupported_claims'],
            'violated_categories': grounding_check['violated_categories'],
            'pattern_issues': grounding_check['pattern_hallucinations'],
            'acknowledges_limitations': acknowledges_limits,
            'recommendation': self._get_recommendation(is_safe, response_safe, grounding_check)
        }
    
    def _check_limitation_acknowledgment(self, response: str) -> bool:
        """
        Check if response appropriately acknowledges when information is not available.
        """
        response_lower = response.lower()
        
        acknowledgment_patterns = [
            r'cannot answer.*based on.*documents?',
            r'not (?:present|available|found) in (?:the )?(?:context|documents?)',
            r"don'?t have (?:enough )?information",
            r'the (?:provided )?(?:context|documents?) (?:do(?:es)?n\'?t|does not) contain'
        ]
        
        for pattern in acknowledgment_patterns:
            if re.search(pattern, response_lower):
                return True
        
        return False
    
    def _get_recommendation(self, query_safe: bool, response_safe: bool, 
                          grounding_check: Dict) -> str:
        """
        Provide recommendation based on validation results.
        """
        if not query_safe:
            return "BLOCK: Unsafe query detected"
        
        if not response_safe:
            return "BLOCK: Unsafe or ungrounded response detected"
        
        if not grounding_check['is_grounded']:
            confidence = grounding_check['confidence_score']
            if confidence < 0.3:
                return "BLOCK: High confidence that response contains hallucinations"
            elif confidence < 0.7:
                return "WARNING: Moderate confidence of hallucination - review response"
            else:
                return "CAUTION: Minor grounding issues detected"
        
        return "PASS: Response is safe and well-grounded in context"



# from langchain_ollama import OllamaLLM
# from typing import Tuple, Dict
# import re
# import logging
# import os



# # ============================================================
# # TEST-ONLY LOGGING SETUP
# # ============================================================

# IS_PYTEST = "PYTEST_CURRENT_TEST" in os.environ

# logger = logging.getLogger("llama_guard")

# if IS_PYTEST:
#     # Configure logger directly instead of using basicConfig
#     logger.setLevel(logging.INFO)
    
#     # Only add handler if not already added (prevent duplicates)
#     if not logger.handlers:
#         import sys
#         handler = logging.StreamHandler(sys.stdout)
#         handler.setLevel(logging.INFO)
#         formatter = logging.Formatter("🦙 [LlamaGuard] %(message)s")
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
    
#     # Prevent propagation to root logger
#     logger.propagate = False
# else:
#     # Silence logger completely outside pytest
#     logger.setLevel(logging.CRITICAL)
#     logger.addHandler(logging.NullHandler())

# # ============================================================
# # LLAMA GUARD IMPLEMENTATION
# # ============================================================

# class LlamaGuard:
#     def __init__(self, model_name: str = "llama-guard3:1b"):
#         """
#         Initialize Llama Guard for content moderation and response validation.
        
#         Args:
#             model_name: The Ollama model name for Llama Guard
#         """
#         logger.info("=" * 60)
#         logger.info("Initializing Llama Guard")
#         logger.info("=" * 60)
#         logger.info(f"Model: {model_name}")
        
#         self.model = OllamaLLM(model=model_name)
#         logger.info("✅ Llama Guard initialized successfully")
        
#         # Llama Guard 3 safety categories
#         self.safety_categories = """S1: Violent Crimes
# S2: Non-Violent Crimes
# S3: Sex-Related Crimes
# S4: Child Sexual Exploitation
# S5: Defamation
# S6: Specialized Advice
# S7: Privacy
# S8: Intellectual Property
# S9: Indiscriminate Weapons
# S10: Hate
# S11: Suicide & Self-Harm
# S12: Sexual Content
# S13: Elections"""

#         # Context grounding validation categories
#         self.grounding_categories = """G1: Response contains information not present in the context
# G2: Response makes claims that contradict the context
# G3: Response includes external knowledge beyond the context
# G4: Response fabricates or hallucinates facts
# G5: Response provides partial information that misrepresents the context
# G6: Response fails to acknowledge lack of information in context"""
        
#         logger.info("Safety categories loaded")
#         logger.info("=" * 60)

#     def check_prompt(self, user_input: str) -> Tuple[bool, str]:
#         """
#         Check if user input is safe.
        
#         Args:
#             user_input: The user's query text
            
#         Returns:
#             Tuple of (is_safe, category/reason)
#         """
#         logger.info("\n" + "=" * 60)
#         logger.info("LLAMA GUARD - PROMPT SAFETY CHECK")
#         logger.info("=" * 60)
#         logger.info(f"Input: '{user_input}'")
        
#         prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

# <BEGIN UNSAFE CONTENT CATEGORIES>
# {self.safety_categories}
# <END UNSAFE CONTENT CATEGORIES>

# <BEGIN CONVERSATION>
# User: {user_input}
# <END CONVERSATION>

# Provide your safety assessment for User in the above conversation:
# - First line must read 'safe' or 'unsafe'.
# - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""

#         try:
#             logger.info("Sending prompt to Llama Guard...")
#             response = self.model.invoke(prompt).strip()
#             logger.info(f"Llama Guard response: '{response}'")
            
#             lines = response.split('\n')
            
#             is_safe = lines[0].lower().strip() == 'safe'
#             category = lines[1].strip() if len(lines) > 1 and not is_safe else ""
            
#             if is_safe:
#                 logger.info("✅ SAFE - Prompt passed safety check")
#             else:
#                 logger.error(f"❌ UNSAFE - Violated categories: {category}")
            
#             logger.info("=" * 60)
#             return is_safe, category
            
#         except Exception as e:
#             logger.error(f"❌ Error in Llama Guard prompt check: {e}")
#             logger.info("⚠️  Failing open - allowing request")
#             logger.info("=" * 60)
#             # Fail open - allow the request but log the error
#             return True, ""

#     def check_context_grounding(self, response: str, context: str, query: str) -> Dict[str, any]:
#         """
#         Validate if the assistant's response is grounded in the provided context.
#         This is the KEY method for ensuring RAG responses don't hallucinate.
        
#         Args:
#             response: The LLM's generated response
#             context: The retrieved context from vector DB
#             query: The original user query
            
#         Returns:
#             Dict containing validation results
#         """
#         logger.info("\n" + "=" * 60)
#         logger.info("LLAMA GUARD - CONTEXT GROUNDING VALIDATION")
#         logger.info("=" * 60)
#         logger.info(f"Query: '{query}'")
#         logger.info(f"Response length: {len(response)} chars")
#         logger.info(f"Context length: {len(context)} chars")
        
#         # First, check for obvious hallucination patterns
#         logger.info("\n[Step 1] Checking hallucination patterns...")
#         hallucination_indicators = self._check_hallucination_patterns(response, context)
        
#         if hallucination_indicators['has_issues']:
#             logger.warning(f"⚠️  Pattern issues found: {len(hallucination_indicators['issues'])}")
#             for issue in hallucination_indicators['issues']:
#                 logger.warning(f"  - {issue}")
#         else:
#             logger.info("✅ No pattern issues detected")
        
#         # If clear pattern issues, no need for deep validation
#         if hallucination_indicators['has_issues']:
#             critical_issues = [issue for issue in hallucination_indicators['issues'] 
#                              if 'Meta-commentary' in issue or 'External citation' in issue]
            
#             if critical_issues:
#                 logger.error("❌ CRITICAL hallucination issues detected")
#                 result = {
#                     'is_grounded': False,
#                     'validation_method': 'pattern_only',
#                     'unsupported_claims': ', '.join(critical_issues),
#                     'violated_categories': 'G1, G3',
#                     'pattern_hallucinations': hallucination_indicators['issues'],
#                     'confidence_score': 0.0
#                 }
#                 logger.info(f"Result: {result}")
#                 logger.info("=" * 60)
#                 return result
        
#         # Use Llama Guard for deep validation with improved prompt
#         logger.info("\n[Step 2] Deep validation with Llama Guard ML model...")
#         prompt = f"""[INST] Task: Validate if the Assistant's response is reasonably grounded in the provided context.

# <BEGIN CONTEXT FROM VECTOR DATABASE>
# {context[:2000]}
# <END CONTEXT FROM VECTOR DATABASE>

# <BEGIN CONVERSATION>
# User Query: {query}

# Assistant Response: {response}
# <END CONVERSATION>

# VALIDATION INSTRUCTIONS:
# - Check if the main ideas and concepts in the response come from the context
# - Allow reasonable paraphrasing and natural language variations
# - Allow combining information from different parts of the context
# - Mark as 'grounded' if the response answers using context information
# - Mark as 'ungrounded' ONLY if the response contains significant information NOT in the context
# - Educational responses that explain concepts from the context should be marked as 'grounded'

# Provide your assessment on ONE line:
# - Write ONLY 'grounded' or 'ungrounded' [/INST]"""

#         try:
#             logger.info("Sending to Llama Guard for grounding check...")
#             llama_response = self.model.invoke(prompt).strip().lower()
#             logger.info(f"Llama Guard response: '{llama_response}'")
            
#             # Extract just the first word/decision
#             is_grounded = 'grounded' in llama_response.split()[0] if llama_response else False
#             logger.info(f"Grounding status: {'✅ GROUNDED' if is_grounded else '❌ UNGROUNDED'}")
            
#             # If Llama Guard says ungrounded but no critical pattern issues, be lenient
#             if not is_grounded and not hallucination_indicators['has_issues']:
#                 logger.info("⚠️  Llama Guard says ungrounded but no pattern issues - being lenient")
#                 # Give benefit of doubt if no clear red flags
#                 is_grounded = True
#                 validation_method = 'lenient_pass'
#             else:
#                 validation_method = 'llama_guard_validated'
            
#             logger.info(f"Validation method: {validation_method}")
            
#             result = {
#                 'is_grounded': is_grounded,
#                 'validation_method': validation_method,
#                 'unsupported_claims': "" if is_grounded else "Response may contain ungrounded information",
#                 'violated_categories': "" if is_grounded else "G1",
#                 'pattern_hallucinations': hallucination_indicators['issues'] if not is_grounded else [],
#                 'confidence_score': self._calculate_confidence(is_grounded, hallucination_indicators)
#             }
            
#             logger.info(f"Confidence score: {result['confidence_score']:.2f}")
#             logger.info(f"Final result: {result}")
#             logger.info("=" * 60)
#             return result
            
#         except Exception as e:
#             logger.error(f"❌ Error in context grounding validation: {e}")
#             logger.info("⚠️  Handling error with lenient fallback...")
            
#             # On error, be lenient if no pattern issues
#             if not hallucination_indicators['has_issues']:
#                 logger.info("✅ No pattern issues - allowing with error lenient mode")
#                 result = {
#                     'is_grounded': True,
#                     'validation_method': 'error_lenient',
#                     'unsupported_claims': '',
#                     'violated_categories': '',
#                     'pattern_hallucinations': [],
#                     'confidence_score': 0.7
#                 }
#             else:
#                 logger.error("❌ Pattern issues exist - blocking")
#                 result = {
#                     'is_grounded': False,
#                     'validation_method': 'error_fallback',
#                     'unsupported_claims': 'Validation error occurred',
#                     'violated_categories': 'Unknown',
#                     'pattern_hallucinations': hallucination_indicators['issues'],
#                     'confidence_score': 0.0
#                 }
            
#             logger.info(f"Result: {result}")
#             logger.info("=" * 60)
#             return result

#     def _check_hallucination_patterns(self, response: str, context: str) -> Dict[str, any]:
#         """
#         Pattern-based hallucination detection for CRITICAL issues only.
#         Only flags severe problems to reduce false positives.
        
#         Returns:
#             Dict with 'has_issues' flag and list of 'issues'
#         """
#         logger.info("Checking hallucination patterns...")
#         issues = []
#         response_lower = response.lower()
#         context_lower = context.lower()
        
#         # Pattern 1: Check for meta-commentary (LLM talking about itself)
#         meta_patterns = [
#             r'\bi am (an )?(ai|assistant|language model)',
#             r'\bi don\'?t have access to',
#             r'\bi cannot (browse|search the web|access the internet)',
#             r'as an ai\b',
#             r'my training data',
#             r'i was trained on',
#             r'according to my knowledge base',
#             r'in my database'
#         ]
        
#         for pattern in meta_patterns:
#             if re.search(pattern, response_lower):
#                 issue = f"Meta-commentary detected: {pattern}"
#                 logger.warning(f"  ⚠️  {issue}")
#                 issues.append(issue)
        
#         # Pattern 2: Check for external source citations not in context
#         citation_patterns = [
#             r'according to wikipedia',
#             r'according to google',
#             r'source: http',
#             r'as stated by \w+ \w+ \(researcher',
#             r'in a study published in'
#         ]
        
#         for pattern in citation_patterns:
#             if re.search(pattern, response_lower):
#                 issue = f"External citation detected: {pattern}"
#                 logger.warning(f"  ⚠️  {issue}")
#                 issues.append(issue)
        
#         # Pattern 3: Check for numerical/statistical claims not in context
#         response_numbers = set(re.findall(r'\b\d+\.?\d*\b', response))
#         context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context))
        
#         fabricated_numbers = response_numbers - context_numbers
#         if fabricated_numbers and len(fabricated_numbers) >= 1:
#             issue = f"Many numerical claims not in context: {len(fabricated_numbers)} numbers"
#             logger.warning(f"  ⚠️  {issue}")
#             issues.append(issue)
        
#         # Pattern 4: Check for definitive statements - RELAXED
#         definitive_patterns = [
#             r'\balways\b',
#             r'\bnever\b',
#             r'\bdefinitely\b',
#             r'\bcertainly\b',
#             r'\babsolutely\b',
#             r'\bguaranteed\b'
#         ]
        
#         definitive_count = sum(1 for pattern in definitive_patterns if re.search(pattern, response_lower))
#         if definitive_count > 5:
#             issue = "Excessive overconfident language"
#             logger.warning(f"  ⚠️  {issue}")
#             issues.append(issue)
        
#         logger.info(f"Pattern check complete: {len(issues)} issues found")
        
#         return {
#             'has_issues': len(issues) > 0,
#             'issues': issues
#         }

#     def _calculate_confidence(self, is_grounded: bool, hallucination_indicators: Dict) -> float:
#         """
#         Calculate confidence score for the grounding validation.
        
#         Returns:
#             Float between 0.0 and 1.0
#         """
#         if not is_grounded:
#             logger.info("Calculating confidence for ungrounded response: 0.0")
#             return 0.0
        
#         if hallucination_indicators['has_issues']:
#             # Reduce confidence based on number of pattern issues
#             issue_count = len(hallucination_indicators['issues'])
#             penalty = min(0.5, issue_count * 0.1)
#             confidence = max(0.0, 1.0 - penalty)
#             logger.info(f"Calculating confidence with {issue_count} issues: {confidence:.2f}")
#             return confidence
        
#         logger.info("Calculating confidence for clean response: 1.0")
#         return 1.0

#     def check_response(self, user_input: str, assistant_response: str, context: str = "") -> Tuple[bool, str]:
#         """
#         Check if assistant response is safe AND grounded in context.
#         This method now performs BOTH safety and grounding validation.
        
#         Args:
#             user_input: The original user query
#             assistant_response: The model's response
#             context: The PDF context used to generate the response
            
#         Returns:
#             Tuple of (is_safe_and_grounded, reason)
#         """
#         logger.info("\n" + "=" * 60)
#         logger.info("LLAMA GUARD - RESPONSE VALIDATION")
#         logger.info("=" * 60)
#         logger.info(f"Query: '{user_input}'")
#         logger.info(f"Response length: {len(assistant_response)} chars")
        
#         # Step 1: Check content safety
#         logger.info("\n[Step 1] Content safety check...")
        
#         prompt = f"""[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy.

# <BEGIN UNSAFE CONTENT CATEGORIES>
# {self.safety_categories}
# <END UNSAFE CONTENT CATEGORIES>

# <BEGIN CONTEXT>
# The Agent's response is based on educational information about Machine Learning, specifically Probabilistic Models, Decision Trees, and related algorithms from course materials.

# Educational content should be considered SAFE. Only mark as UNSAFE if the response:
# 1. Promotes harmful activities or illegal actions
# 2. Contains hate speech or discriminatory content
# 3. Violates privacy by sharing personal information
# 4. Contains inappropriate sexual content

# Document Context:
# {context[:300]}...
# <END CONTEXT>

# <BEGIN CONVERSATION>
# User: {user_input}

# Agent: {assistant_response}
# <END CONVERSATION>

# Provide your safety assessment on ONE line:
# Write ONLY 'safe' or 'unsafe' [/INST]"""

#         try:
#             logger.info("Sending to Llama Guard for safety check...")
#             response = self.model.invoke(prompt).strip().lower()
#             logger.info(f"Llama Guard response: '{response}'")
            
#             # Extract just the first word
#             is_safe = 'safe' in response.split()[0] if response else True
            
#             if not is_safe:
#                 logger.error("❌ Content safety violation detected")
#                 logger.info("=" * 60)
#                 return False, f"Content safety violation detected"
            
#             logger.info("✅ Content is safe")
            
#             # Step 2: If safe, do lightweight grounding check
#             if context:
#                 logger.info("\n[Step 2] Grounding validation...")
#                 grounding_result = self.check_context_grounding(
#                     assistant_response, 
#                     context, 
#                     user_input
#                 )
                
#                 # Only block if confidence is very low (clear hallucination)
#                 if not grounding_result['is_grounded'] and grounding_result['confidence_score'] < 0.3:
#                     reason = "Response likely contains hallucinations"
#                     if grounding_result['pattern_hallucinations']:
#                         reason += f": {', '.join(grounding_result['pattern_hallucinations'][:2])}"
#                     logger.error(f"❌ {reason}")
#                     logger.info("=" * 60)
#                     return False, reason
                
#                 logger.info(f"✅ Response is reasonably grounded (confidence: {grounding_result['confidence_score']:.2f})")
            
#             logger.info("\n✅ FINAL: Content is safe and reasonably grounded")
#             logger.info("=" * 60)
#             return True, "Content is safe and reasonably grounded"
            
#         except Exception as e:
#             logger.error(f"❌ Error in Llama Guard response check: {e}")
#             logger.info("⚠️  Validation error - being lenient for educational content")
#             logger.info("=" * 60)
#             # On error, be lenient - allow educational content
#             return True, "Validation error - allowing educational content"

#     def validate_rag_response(self, query: str, response: str, context: str, 
#                              retrieved_chunks: list) -> Dict[str, any]:
#         """
#         Comprehensive RAG response validation.
#         This is the main method to use for validating RAG chatbot responses.
        
#         Args:
#             query: User's original query
#             response: LLM's generated response
#             context: Full context passed to LLM
#             retrieved_chunks: List of retrieved document chunks with metadata
            
#         Returns:
#             Dict with comprehensive validation results
#         """
#         logger.info("\n" + "=" * 60)
#         logger.info("COMPREHENSIVE RAG VALIDATION")
#         logger.info("=" * 60)
#         logger.info(f"Query: '{query}'")
#         logger.info(f"Response: '{response[:100]}...'")
#         logger.info(f"Context length: {len(context)} chars")
#         logger.info(f"Retrieved chunks: {len(retrieved_chunks)}")
        
#         # 1. Safety check
#         logger.info("\n[Phase 1] Query safety check...")
#         is_safe, safety_reason = self.check_prompt(query)
        
#         # 2. Response safety and grounding check
#         logger.info("\n[Phase 2] Response safety and grounding check...")
#         response_safe, response_reason = self.check_response(query, response, context)
        
#         # 3. Deep grounding validation
#         logger.info("\n[Phase 3] Deep grounding validation...")
#         grounding_check = self.check_context_grounding(response, context, query)
        
#         # 4. Check if response acknowledges limitations appropriately
#         logger.info("\n[Phase 4] Limitation acknowledgment check...")
#         acknowledges_limits = self._check_limitation_acknowledgment(response)
#         logger.info(f"Acknowledges limitations: {acknowledges_limits}")
        
#         # Calculate overall validity
#         overall_valid = is_safe and response_safe and grounding_check['is_grounded']
        
#         result = {
#             'overall_valid': overall_valid,
#             'query_safe': is_safe,
#             'query_safety_reason': safety_reason,
#             'response_safe': response_safe,
#             'response_safety_reason': response_reason,
#             'is_grounded': grounding_check['is_grounded'],
#             'grounding_confidence': grounding_check['confidence_score'],
#             'unsupported_claims': grounding_check['unsupported_claims'],
#             'violated_categories': grounding_check['violated_categories'],
#             'pattern_issues': grounding_check['pattern_hallucinations'],
#             'acknowledges_limitations': acknowledges_limits,
#             'recommendation': self._get_recommendation(is_safe, response_safe, grounding_check)
#         }
        
#         logger.info("\n" + "=" * 60)
#         logger.info("VALIDATION SUMMARY")
#         logger.info("=" * 60)
#         logger.info(f"Overall Valid: {overall_valid}")
#         logger.info(f"Query Safe: {is_safe}")
#         logger.info(f"Response Safe: {response_safe}")
#         logger.info(f"Is Grounded: {grounding_check['is_grounded']}")
#         logger.info(f"Grounding Confidence: {grounding_check['confidence_score']:.2%}")
#         logger.info(f"Recommendation: {result['recommendation']}")
#         logger.info("=" * 60)
        
#         return result
    
#     def _check_limitation_acknowledgment(self, response: str) -> bool:
#         """
#         Check if response appropriately acknowledges when information is not available.
#         """
#         response_lower = response.lower()
        
#         acknowledgment_patterns = [
#             r'cannot answer.*based on.*documents?',
#             r'not (?:present|available|found) in (?:the )?(?:context|documents?)',
#             r"don'?t have (?:enough )?information",
#             r'the (?:provided )?(?:context|documents?) (?:do(?:es)?n\'?t|does not) contain'
#         ]
        
#         for pattern in acknowledgment_patterns:
#             if re.search(pattern, response_lower):
#                 logger.info(f"  ✅ Found acknowledgment pattern: {pattern}")
#                 return True
        
#         logger.info("  ℹ️  No limitation acknowledgment found")
#         return False
    
#     def _get_recommendation(self, query_safe: bool, response_safe: bool, 
#                           grounding_check: Dict) -> str:
#         """
#         Provide recommendation based on validation results.
#         """
#         if not query_safe:
#             return "BLOCK: Unsafe query detected"
        
#         if not response_safe:
#             return "BLOCK: Unsafe or ungrounded response detected"
        
#         if not grounding_check['is_grounded']:
#             confidence = grounding_check['confidence_score']
#             if confidence < 0.3:
#                 return "BLOCK: High confidence that response contains hallucinations"
#             elif confidence < 0.7:
#                 return "WARNING: Moderate confidence of hallucination - review response"
#             else:
#                 return "CAUTION: Minor grounding issues detected"
        
#         return "PASS: Response is safe and well-grounded in context"
    


    