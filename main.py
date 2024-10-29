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
    
    # def transfer_to_agent_b():
    #     return agent_b
    agent = Agent(
        model="microsoft/Phi-3.5-mini-instruct",
        name="Programmer Agent",
        instructions="You are a reasoning assistant. When presented with a problem, please break down the solution step by step, explaining your thought process clearly.",
        # functions=[],
    )
    stream = True
    messages = []

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables={},
            stream=stream,
        )

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent = response.agent
    # agent_a = Agent(
    #     name="Agent A",
    #     instructions="You are a helpful agent.",
    #     functions=[transfer_to_agent_b],
    # )

    # agent_b = Agent(
    #     name="Agent B",
    #     instructions="Only speak in Haikus.",
    # )

    # response = client.run(
    #     agent=agent_a,
    #     messages=[{"role": "user", "content": "I want to talk to agent B."}],
    # )

    # print(response.messages[-1]["content"])
    

if __name__ == "__main__":
    start()