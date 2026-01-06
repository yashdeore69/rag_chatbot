from langchain_ollama import OllamaLLM
from typing import Tuple

class LlamaGuard:
    def __init__(self, model_name: str = "llama-guard3:1b"):
        """
        Initialize Llama Guard for content moderation.
        
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

    def check_response(self, user_input: str, assistant_response: str, context: str = "") -> Tuple[bool, str]:
        """
        Check if assistant response is safe, considering it's based on PDF documents.
        
        Args:
            user_input: The original user query
            assistant_response: The model's response
            context: The PDF context used to generate the response
            
        Returns:
            Tuple of (is_safe, category/reason)
        """
        # Enhanced prompt that tells Llama Guard this is educational/informational content from documents
        prompt = f"""[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{self.safety_categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONTEXT>
IMPORTANT: The Agent's response is based on information retrieved from legitimate PDF documents provided by the user. The Agent is acting as an information retrieval system, presenting factual content from these documents.

Academic, educational, technical, medical, and informational content from legitimate sources should be considered SAFE, even if it discusses sensitive topics, as long as:
1. The information is factual and educational in nature
2. The response is directly answering the user's question based on the provided documents
3. The response is not promoting harmful activities or providing instructions for illegal actions
4. The response is not violating privacy by sharing personal information not present in the documents

Document Context Preview:
{context[:500]}...
<END CONTEXT>

<BEGIN CONVERSATION>
User: {user_input}

Agent: {assistant_response}
<END CONVERSATION>

Provide your safety assessment for Agent in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.
- Remember: Educational/informational content from documents is SAFE. [/INST]"""

        try:
            response = self.model.invoke(prompt).strip()
            lines = response.split('\n')
            
            is_safe = lines[0].lower().strip() == 'safe'
            category = lines[1].strip() if len(lines) > 1 and not is_safe else ""
            
            return is_safe, category
        except Exception as e:
            print(f"Error in Llama Guard response check: {e}")
            # Fail open - allow the response but log the error
            return True, ""