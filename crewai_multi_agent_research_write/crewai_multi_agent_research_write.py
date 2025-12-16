"""
CrewAI Multi-Agent Research & Writing System
Orchestrates AI agents for automated research and content creation.
"""

import os
from getpass import getpass
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process, LLM

# Set up the environment variable for SERPER_API_KEY
# Load API key from environment or prompt user
if __name__ == "__main__":
    if 'SERPER_API_KEY' not in os.environ:
        os.environ['SERPER_API_KEY'] = getpass("Enter your SERPER_API_KEY: ")

    # Initialize the SerperDevTool
    search_tool = SerperDevTool()
    print(type(search_tool))

    # Test the search tool with a sample query
    search_query = "Latest Breakthroughs in machine learning"
    search_results = search_tool.run(query=search_query)

    # Print the results
    print(f"Search Results for '{search_query}':\n{search_results}")

    # Initialize the LLM with the specified parameters
    llm = LLM(
            model="watsonx/meta-llama/llama-3-3-70b-instruct",
            base_url="https://us-south.ml.cloud.ibm.com",
            project_id="skills-network",
            max_tokens=2000,
    )

    # Define the Research Agent
    research_agent = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge information and insights on any subject with comprehensive analysis',
    backstory="""You are an expert researcher with extensive experience in gathering, analyzing, and synthesizing information across multiple domains. 
    Your analytical skills allow you to quickly identify key trends, separate fact from opinion, and produce insightful reports on any topic. 
    You excel at finding reliable sources and extracting valuable information efficiently.""",
    verbose=True,
    allow_delegation=False,
    llm = llm,
    tools=[SerperDevTool()]
    )

    # Define the Writer Agent
    writer_agent = Agent(
    role='Tech Content Strategist',
    goal='Craft well-structured and engaging content based on research findings',
    backstory="""You are a skilled content strategist known for translating 
    complex topics into clear and compelling narratives. Your writing makes 
    information accessible and engaging for a wide audience.""",
    verbose=True,
    llm = llm,
    allow_delegation=True
    )

    # Define the Research Task for the research agent
    research_task = Task(
    description="Analyze the major {topic}, identifying key trends and technologies. Provide a detailed report on their potential impact.",
    agent=research_agent,
    expected_output="A detailed report on {topic}, including trends, emerging technologies, and their impact."
    )

    # Define the Writer Task for the writer agent
    writer_task = Task(
    description="Create an engaging blog post based on the research findings about {topic}. Tailor the content for a tech-savvy audience, ensuring clarity and interest.",
    agent=writer_agent,
    expected_output="A 4-paragraph blog post on {topic}, written clearly and engagingly for tech enthusiasts."
    )

    # Initialize the Crew with the specified parameters
    crew = Crew(
        agents=[research_agent, writer_agent],
        tasks=[research_task, writer_task],
        process=Process.sequential,
        verbose=True 
    )

    # Initialize Kickoff
    try:
        result = crew.kickoff(inputs={"topic": "Latest Generative AI breakthroughs"})
    except Exception as e:
        print(f"Error during crew execution: {e}")
        raise

    # Output final content of last agent
    final_output = result.raw
    print("Final output:", final_output)

    # View outputs of each task consecutively 
    tasks_outputs = result.tasks_output

    # View output for research task object for both description and agent output
    print("Task Description", tasks_outputs[0].description)
    print("Output of research task ",tasks_outputs[0])

    # View output for writer task object for both description and agent output
    print("Writer task description:", tasks_outputs[1].description)
    print(" \nOutput of writer task:", tasks_outputs[1].raw)

    # View agent of each task
    print("We can get the agent for researcher task:  ",tasks_outputs[0].agent)
    print("We can get the agent for the writer task: ",tasks_outputs[1].agent)

    ## Detailed metrics on performance and cost
    token_count = result.token_usage.total_tokens
    prompt_tokens = result.token_usage.prompt_tokens
    completion_tokens = result.token_usage.completion_tokens

    print(f"Total tokens used: {token_count}")
    print(f"Prompt tokens: {prompt_tokens} (used for instructions to the model)")
    print(f"Completion tokens: {completion_tokens} (generated in response)")
