import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool

async def force_after_pattern_agent_example():
    """
    RequirementAgent with Force-After Pattern
    
    Demonstrates force_after parameter to mandate reasoning after every tool call.
    Same query - but now with enforced reflection between tool uses.
    """

    llm = ChatModel.from_name("watsonx:meta-llama/llama-4-maverick-17b-128e-instruct-fp8", ChatModelParameters(temperature=0))
    
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

Your methodology:
1. Analyze the threat landscape systematically
2. Research authoritative sources when available
3. Provide comprehensive risk assessment with actionable recommendations
4. Focus on practical, implementable security measures"""
    
    # RequirementAgent with reasoning + research capability
    agent = RequirementAgent(
        llm=llm,
        tools=[ThinkTool(), WikipediaTool()],  # Thinking + Research
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,  # Thinking required first
                force_after=Tool,  # Force reasoning after every tool call
                min_invocations=1,  # At least 1 invocation
                max_invocations=5,  # Max of 5 invocations
                consecutive_allowed=False  # No consecutive thinking
            )
        ] 
    )
    
    query = """Analyze the cybersecurity risks of quantum computing for financial institutions. What are the main threats, timeline for concern, and recommended preparation strategies?"""
    
    result = await agent.run(query)
    print(f"\nReasoning + Research Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await force_after_pattern_agent_example()

if __name__ == "__main__":
    asyncio.run(main())