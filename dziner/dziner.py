import langchain
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentType
from langchain_core.prompts import MessagesPlaceholder
from .prompts import prefix, suffix


class dziner:
    def __init__(
        self,
        tools,
        agent_model="text-davinci-003",
        temp=0.1,
        get_cost=False,
        max_iterations=40,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        **kwargs,
    ):
        if agent_model.startswith("gpt-3.5-turbo") or agent_model.startswith("gpt-4"):
            self.agent_model = langchain.chat_models.ChatOpenAI(
                temperature=temp,
                model_name=agent_model,
                request_timeout=1000,
                max_tokens=4096,
            )
        self.get_cost = get_cost
        self.max_iterations = max_iterations

        # Initialize agent
        memory = ConversationBufferMemory(memory_key="chat_history",
                                            input_key='input',
                                            output_key="output",
                                            return_messages=True)
        
        self.verbose = kwargs.get('verbose', False)
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        self.agent = initialize_agent(
            tools=tools,
            llm=self.agent_model,
            agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=self.verbose,
            memory=memory,
            early_stopping_method='generate',
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": prefix,
                # "system_message": SystemMessage(content=message),
                "suffix": suffix,
                "memory_prompts": [chat_history],
                "input_variables": [
                    "input",
                    "agent_scratchpad",
                    "chat_history"
                ],
            },
            return_intermediate_steps=True,
            max_iterations=40
        )

    def run(self, prompt):
        with get_openai_callback() as cb:
            result = self.agent_chain.run(input=prompt)
        if self.get_cost:
            print(cb)
        return result
