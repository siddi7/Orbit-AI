import requests
import json
import logging
import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ORBIT-AI")

class SharedMemory:
    def __init__(self):
        self.data: Dict[str, Any] = {}          # Store DataFrames or data objects
        self.plan: List[str] = []               # List of planned steps
        self.insights: List[str] = []           # Key findings
        self.logs: List[Dict[str, Any]] = []    # Execution logs
        self.reports: Dict[str, str] = {}       # Final or intermediate reports
        self.figures: Dict[str, Any] = {}       # Plotly figures or image paths
        self.context: Dict[str, Any] = {}       # General context/metadata

    def log(self, agent_name: str, message: str):
        timestamp = datetime.datetime.now().isoformat()
        entry = {"timestamp": timestamp, "agent": agent_name, "message": message}
        self.logs.append(entry)
        logger.info(f"[{agent_name}] {message}")

    def to_json(self):
        # Helper to dump serializable parts of memory
        return json.dumps({
            "plan": self.plan,
            "insights": self.insights,
            "logs": self.logs,
            "context": self.context
        }, indent=2)

class BaseAgent:
    def __init__(self, name: str, role: str, model: str = "llama3"):
        self.name = name
        self.role = role
        self.model = model

    def run(self, shared_memory: SharedMemory):
        """
        Main execution method for the agent.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement run()")

    def ask_llm(self, prompt: str, system_prompt: str = None, json_mode: bool = False) -> str:
        """
        Interacts with the local Ollama instance.
        """
        if system_prompt is None:
            system_prompt = f"You are {self.name}, a {self.role} in the ORBIT-AI system. Perform your task efficiently."

        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            # Fallback for demo purposes if Ollama is not running
            logger.error(f"Ollama connection failed: {e}")
            return f"Error: Could not connect to Ollama ({str(e)}). Please ensure Ollama is running."
