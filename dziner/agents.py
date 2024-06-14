import langchain
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentType
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from .prompts import prefix, suffix


class dZiner:
    def __init__(
        self,
        tools,
        property,
        model="text-davinci-003",
        temp=0.1,
        get_cost=False,
        max_iterations=40,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        **kwargs,
    ):
        self.property = property
        # keeping the variables the same so they are updated later in langchain.
        formatted_suffix = suffix.format(property=self.property,
                                         tool_desc="{tool_desc}",
                                        input="{input}",
                                        agent_scratchpad="{agent_scratchpad}")
        # formatted_prefix = prefix.format(property=self.property,
        #                                  tool_names="{tool_names}")
        self.suffix = kwargs.get('suffix', formatted_suffix)
        self.prefix = kwargs.get('prefix', prefix)

        self.model = model
        if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            self.model = ChatOpenAI(
                temperature=temp,
                model_name=model,
                request_timeout=1000,
                max_tokens=4096,
            )
        self.get_cost = get_cost
        self.max_iterations = max_iterations

        # Initialize agent
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key='input',
            output_key="output",
            return_messages=True
        )

        self.verbose = kwargs.get('verbose', False)
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        self.agent = initialize_agent(
            tools=tools,
            llm=self.model,
            agent_type=agent_type,
            verbose=self.verbose,
            memory=memory,
            early_stopping_method='generate',
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": self.prefix,
                "suffix": self.suffix,
                "memory_prompts": [chat_history],
                "input_variables": [
                    "input",
                    "agent_scratchpad",
                    "chat_history"
                ],
            },
            return_intermediate_steps=True,
            max_iterations=max_iterations
        )

    def __call__(self, prompt):
        with get_openai_callback() as cb:
            result = self.agent.invoke({"input":prompt})
        if self.get_cost:
            print(cb)
        return result