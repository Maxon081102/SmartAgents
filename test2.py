import hydra
import logging

from omegaconf import OmegaConf, DictConfig
from openai import OpenAI
from swarm import Swarm, Agent
from swarm.repl.repl import pretty_print_messages, process_and_print_streaming_response
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='base_model')
def start(cfg: DictConfig):
    openai_api_key = cfg["key"]
    openai_api_base = cfg["host"]

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    client = Swarm(client=client)
    
    def transfer_to_agent_b():
        return agent_b
    
    agent_a = Agent(
        model=cfg['model'],
        name="Agent A",
        instructions="You are a helpful agent.",
        functions=[transfer_to_agent_b],
    )

    agent_b = Agent(
        model=cfg['model'],
        name="Agent B",
        instructions="Only speak in Haikus.",
    )

    response = client.run(
        agent=agent_a,
        messages=[{"role": "user", "content": "I want to talk to agent B."}],
    )

    print(response.messages[-1]["content"])
    

if __name__ == "__main__":
    start()