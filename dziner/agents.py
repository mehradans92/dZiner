import langchain
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentType
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from .prompts import SUFFIX, PREFIX, FORMAT_INSTRUCTIONS
import os


class dZiner:
    def __init__(
        self,
        tools,
        property,
        model="gpt-4o",
        temp=0.1,
        get_cost=False,
        max_iterations=40,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        n_design_iterations=1,
        early_stopping_method="generate",
        **kwargs
    ):
        self.property = property
        self.n_design_iterations = n_design_iterations
        self.early_stopping_method = early_stopping_method
        # keeping the variables the same so they are updated later in langchain.
        self.suffix = kwargs.get('suffix',
                                 SUFFIX.format(property=self.property,
                                         tool_desc="{tool_desc}",
                                        input="{input}",
                                        agent_scratchpad="{agent_scratchpad}",
                                        n_design_iterations=self.n_design_iterations))
        self.prefix = kwargs.get('prefix',
                                 PREFIX.format(property=self.property))
        self.format_instructions = kwargs.get('format_instructions',
                                                FORMAT_INSTRUCTIONS.format(tool_names="{tool_names}"))
        self.callbacks = kwargs.get('callbacks',
                                                None)

        if type(model) == str:
            if model.startswith("gpt"):
                self.model = ChatOpenAI(
                    temperature=temp,
                    model_name=model,
                    request_timeout=1000,
                    max_tokens=4096,
                )
            elif model.startswith("claude"):
               self.model = ChatAnthropic(model=model,
                                          temperature=temp,
                                          max_tokens=8192, api_key=os.environ['ANTHROPIC_API_KEY'])
            else:
                # TODO: Implement support for non-OpenAI models
                raise NotImplementedError("None-OpenAI models are not implemented yet.")
        else: 
            self.model = model
        self.get_cost = get_cost
        self.max_iterations = max_iterations

        # Initialize agent
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key='input',
            output_key="output",
            s_messages=True,
            return_messages=True
        )

        self.verbose = kwargs.get('verbose', False)
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        self.tools = tools
        self.agent = initialize_agent(
            tools=tools,
            llm=self.model,
            agent_type=agent_type,
            verbose=self.verbose,
            memory=memory,
            stop=["\nAction:", "\nObservation:", "\nFinal Answer:", "\nAnswer:"],
            early_stopping_method=self.early_stopping_method,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message" : """You are dZiner, an expert chemist AI assistant. You must:
                            1. Always check chat history before responding
                            2. Use information from previous messages (like names, preferences, or context)
                            3. Maintain consistent knowledge of previous interactions
                            4. Never reintroduce yourself if you've already done so
                            5. Answer questions based on the accumulated context""",
                "prefix": self.prefix,
                "suffix": self.suffix,
                "format_instructions": self.format_instructions,
                "memory_prompts": [chat_history],
                "input_variables": [
                    "input",
                    "agent_scratchpad",
                    "chat_history"
                ],
            },
            return_intermediate_steps=True,
            max_iterations=max_iterations,
            callbacks= self.callbacks
        )

    def __call__(self, prompt):
        with get_openai_callback() as cb:
            tool_desc = [tool.description for tool in self.tools]
            result = self.agent.invoke({"input":prompt, "tool_desc": tool_desc})
            # print("After response chat history:", self.agent.memory.chat_memory.messages)
        if self.get_cost:
            print(cb)
        return result
    
    async def iter(self, prompt):
        """
        This method allows controlled iteration through the agent's response, which is useful for handling
        intermediate steps such as feedback from the user.
        """
        tool_desc = [tool.description for tool in self.tools]
        input_data = {"input": prompt, "tool_desc": tool_desc}
        async for step in self.agent.aiter(input_data):
            yield step