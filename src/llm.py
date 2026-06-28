import json
import requests
import re
from google import genai
from google.genai import types
from src import config

def get_active_config():
    """Retrieve active LLM configurations (either from streamlit session state or config file)"""
    try:
        import streamlit as st
        if st.runtime.exists():
            provider = st.session_state.get("llm_provider", config.DEFAULT_PROVIDER)
            ollama_url = st.session_state.get("ollama_url", config.OLLAMA_BASE_URL)
            ollama_model = st.session_state.get("ollama_model", config.OLLAMA_MODEL)
            ollama_embed = st.session_state.get("ollama_embed", config.OLLAMA_EMBED_MODEL)
            gemini_key = st.session_state.get("gemini_key", config.GEMINI_API_KEY)
            gemini_model = st.session_state.get("gemini_model", config.GEMINI_MODEL)
            gemini_embed = st.session_state.get("gemini_embed", config.GEMINI_EMBED_MODEL)
            return {
                "provider": provider,
                "ollama_url": ollama_url,
                "ollama_model": ollama_model,
                "ollama_embed": ollama_embed,
                "gemini_key": gemini_key,
                "gemini_model": gemini_model,
                "gemini_embed": gemini_embed,
            }
    except Exception:
        pass
    
    return {
        "provider": config.DEFAULT_PROVIDER,
        "ollama_url": config.OLLAMA_BASE_URL,
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_embed": config.OLLAMA_EMBED_MODEL,
        "gemini_key": config.GEMINI_API_KEY,
        "gemini_model": config.GEMINI_MODEL,
        "gemini_embed": config.GEMINI_EMBED_MODEL,
    }

def test_connection() -> tuple[bool, str]:
    """Test connection to the configured LLM provider. Returns (success, message)."""
    cfg = get_active_config()
    if cfg["provider"] == "gemini":
        if not cfg["gemini_key"]:
            return False, "Gemini API key is not configured."
        try:
            client = genai.Client(api_key=cfg["gemini_key"])
            response = client.models.generate_content(
                model=cfg["gemini_model"],
                contents="Ping",
                config=types.GenerateContentConfig(max_output_tokens=10)
            )
            return True, f"Successfully connected to Gemini using model: {cfg['gemini_model']}"
        except Exception as e:
            return False, f"Failed to connect to Gemini: {str(e)}"
    else:
        url = f"{cfg['ollama_url']}/api/generate"
        try:
            # We check if Ollama is running by asking for model generation
            response = requests.post(
                url,
                json={"model": cfg["ollama_model"], "prompt": "Ping", "stream": False, "options": {"num_predict": 10}},
                timeout=45
            )
            if response.status_code == 200:
                return True, f"Successfully connected to Ollama using model: {cfg['ollama_model']}"
            else:
                return False, f"Ollama returned status code {response.status_code}. Response: {response.text}"
        except requests.exceptions.RequestException as e:
            return False, f"Failed to connect to Ollama at {cfg['ollama_url']}: {str(e)}"

def generate_text(prompt: str, system_instruction: str = None) -> str:
    """Generate text from prompt using active provider."""
    cfg = get_active_config()
    if cfg["provider"] == "gemini":
        if not cfg["gemini_key"]:
            raise ValueError("Gemini API key is not configured.")
        client = genai.Client(api_key=cfg["gemini_key"])
        
        gen_config = None
        if system_instruction:
            gen_config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0)
        else:
            gen_config = types.GenerateContentConfig(temperature=0.0)
            
        response = client.models.generate_content(
            model=cfg["gemini_model"],
            contents=prompt,
            config=gen_config
        )
        return response.text
    else:
        url = f"{cfg['ollama_url']}/api/generate"
        payload = {
            "model": cfg["ollama_model"],
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        if system_instruction:
            payload["system"] = system_instruction
            
        try:
            response = requests.post(url, json=payload, timeout=90)
            if response.status_code == 404:
                raise ValueError(f"Model '{cfg['ollama_model']}' was not found in your Ollama server. Please wait for the background download to complete.")
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
                raise ValueError(f"Model '{cfg['ollama_model']}' was not found in your Ollama server. Please wait for the background download to complete.")
            raise e

def clean_json_string(text: str) -> str:
    """Helper to strip markdown backticks and clean a string so it can be parsed as JSON"""
    # Remove markdown code blocks if present
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def generate_json(prompt: str, system_instruction: str = None) -> dict | list:
    """Generate and parse JSON from the LLM. Retries once with clean prompt on JSON parse failure."""
    cfg = get_active_config()
    
    # Try once
    try:
        response_text = generate_text(prompt, system_instruction)
        cleaned = clean_json_string(response_text)
        return json.loads(cleaned)
    except Exception as e:
        # If it fails, retry once with an explicit request for clean JSON
        retry_prompt = f"{prompt}\n\nCRITICAL: The previous output failed to parse as valid JSON. Ensure you return ONLY a raw JSON format. No markdown blocks, no extra text."
        response_text = generate_text(retry_prompt, system_instruction)
        cleaned = clean_json_string(response_text)
        return json.loads(cleaned)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using the active provider."""
    if not texts:
        return []
        
    cfg = get_active_config()
    if cfg["provider"] == "gemini":
        if not cfg["gemini_key"]:
            raise ValueError("Gemini API key is not configured.")
        client = genai.Client(api_key=cfg["gemini_key"])
        
        # We fetch embeddings in batch or individually
        embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model=cfg["gemini_embed"],
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings
    else:
        url = f"{cfg['ollama_url']}/api/embed"
        payload = {
            "model": cfg["ollama_embed"],
            "input": texts
        }
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["embeddings"]
