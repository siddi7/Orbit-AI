from .core import SharedMemory, logger
from .agents import PlannerAgent, DataScoutAgent, DataScienceAgent, MLAgent, CriticAgent, SynthesizerAgent

class ExecutionAgent:
    def __init__(self):
        self.memory = SharedMemory()
        self.agents = [
            PlannerAgent(),
            DataScoutAgent(),
            DataScienceAgent(),
            MLAgent(),
            CriticAgent(),
            SynthesizerAgent()
        ]

    def run(self, user_goal: str):
        self.memory.context["goal"] = user_goal
        
        for agent in self.agents:
            logger.info(f"--- Starting {agent.name} ---")
            yield agent.name, "Running..."
            try:
                agent.run(self.memory)
                yield agent.name, "Completed"
            except Exception as e:
                logger.error(f"Error in {agent.name}: {e}")
                self.memory.log(agent.name, f"Error: {e}")
                yield agent.name, "Failed"
        
        return self.memory
