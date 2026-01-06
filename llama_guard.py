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
        if fabricated_numbers and len(fabricated_numbers) > 5:
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