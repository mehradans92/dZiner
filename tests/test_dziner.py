import pytest
from langchain.agents import Tool
from langchain.agents.agent_types import AgentType
from fake_llm import FakeChatModel
from dziner import prompts

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import the dZiner class and prompts
from dziner import dZiner

def test_dziner():
    # Create mock tool function
    def test_tool_func(input_str):
        return "Test tool output for input: " + input_str

    # Create a mock tool
    test_tool = Tool(
        name="TestTool",
        func=test_tool_func,
        description="This is a test tool that echoes the input."
    )

    # Define fake responses for the chat model
    responses = [
        "Thought: I need to find the best SMILES.\nAction: TestTool\nAction Input: Generate new SMILES",
        "Observation: Generated SMILES: CCO\nThought: I have the final answer.\nFinal Answer: {\"Iteration\": 1, \"SMILES\": \"CCO\", \"Modification\": \"Added an ethyl group\", \"Docking Score\": -9.5, \"Validity\": \"Valid\"}"
    ]
    fake_chat_model = FakeChatModel(responses=responses)

    initial_molecule = "CN1CCN(C2=C(NC(C3=CC=CC=C3Cl)=O)C=C([N+]([O-])=O)C=C2)CC1"
    Human_prompt = f"Make changes to {initial_molecule} so it will have a larger binding affinity against WDR5 target. Suggest 1 new candidate."

    RetrievalQA_prompt = "Provide design guidelines based on the latest research."

    tools = [test_tool]
    tool_names = [tool.name for tool in tools]
    tool_desc = [tool.description for tool in tools]

    property = "Binding affinity"
    n_design_iterations = 1

    SUFFIX = prompts.SUFFIX.format(
        property=property,
        n_design_iterations=n_design_iterations,
        input="{input}",
        agent_scratchpad="{agent_scratchpad}",
        tool_desc="{tool_desc}"
    )

    FORMAT_INSTRUCTIONS = prompts.FORMAT_INSTRUCTIONS.format(
        tool_names=", ".join(tool_names),
        n_design_iterations=n_design_iterations
    )

    dziner_instance = dZiner(
        tools=tools,
        property=property,
        model=fake_chat_model,
        temp=0.4,
        get_cost=False,
        max_iterations=2, 
        n_design_iterations=n_design_iterations,
        suffix=SUFFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True
    )

    # Prepare the input data
    input_data = {
        "input": Human_prompt,
        "prompt": RetrievalQA_prompt,
        "tools": tools,
        "tool_names": tool_names,
        "tool_desc": tool_desc
    }

    result = dziner_instance.agent.invoke(input_data)

    assert result is not None
    assert "\"SMILES\": \"CCO\"" in str(result)
