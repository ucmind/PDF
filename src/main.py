import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool
from dotenv import load_dotenv

# Load API keys from .env (Requires OPENAI_API_KEY and SERPER_API_KEY)
# Note: CrewAI will automatically use the OPENAI_MODEL_NAME from .env if set.
load_dotenv()

def run_lite_crew():
    print("=== Starting SaaS Launchpad Crew (Lite Version) ===")
    
    # 1. Get User Input
    idea = input("\nPlease enter your project idea (e.g., An AI Line Bot that tracks my expenses): ")
    if not idea:
        print("No idea entered. Exiting...")
        return

    # 2. Initialize Core Tools
    search_tool = SerperDevTool()
    file_writer = FileWriterTool()

    # 3. Define Agents
    
    # Agent A: The Analyst
    # Responsible for breaking down the abstract idea into concrete requirements.
    analyst = Agent(
        role='Lead Product Analyst',
        goal='Analyze the user idea and produce a clear Product Requirements Document (PRD).',
        backstory=(
            "You are an expert at translating vague ideas into logical development "
            "documentation and identifying potential market risks."
        ),
        tools=[search_tool, file_writer],
        verbose=True,
        memory=True
    )

    # Agent B: The Resource Hunter
    # Responsible for finding existing tools to prevent reinventing the wheel.
    resource_hunter = Agent(
        role='Tech Resource Scout',
        goal='Find the most suitable Python libraries, APIs, and open-source projects.',
        backstory=(
            "You are deeply familiar with the Python ecosystem and GitHub. "
            "You always find existing tools and libraries to accelerate development."
        ),
        tools=[search_tool, file_writer],
        verbose=True,
        memory=True
    )

    # Agent C: The Architect
    # Responsible for synthesizing the spec and stack into a code structure.
    architect = Agent(
        role='Senior Technical Architect',
        goal='Produce the core MVP code structure based on analysis and resources.',
        backstory=(
            "You excel at rapidly building functional MVPs. "
            "You focus on code simplicity, modularity, and best practices."
        ),
        tools=[file_writer], # The architect focuses on writing code, not searching.
        verbose=True,
        memory=True
    )

    # 4. Define Tasks (Consolidated into 3 core steps)

    # Task 1: Specification & Analysis
    task_analysis = Task(
        description=(
            f"Analyze the user's idea: '{idea}'.\n"
            "1. Define 3-5 core features (MVP scope).\n"
            "2. Identify potential technical challenges.\n"
            "3. Use search_tool to find similar products and list 2 competitors.\n"
            "Write the result to 'lite_output/1_spec.md'."
        ),
        expected_output="A Markdown document containing the feature list, technical challenges, and competitor analysis.",
        agent=analyst
    )

    # Task 2: Tech Stack Selection
    task_resources = Task(
        description=(
            "Based on the analyst's spec, recommend the technology stack.\n"
            "1. Search and list the best 3 Python libraries (include 'pip install' commands).\n"
            "2. If external APIs (e.g., OpenAI, Line, Weather) are required, list recommended providers.\n"
            "Write the result to 'lite_output/2_tech_stack.md'."
        ),
        expected_output="A Markdown document containing the Python package list and API recommendations.",
        agent=resource_hunter
    )

    # Task 3: Skeleton Code Generation
    task_coding = Task(
        description=(
            "Based on the spec and tech stack, write the core 'main.py'.\n"
            "1. This is a 'Skeleton', not the full product.\n"
            "2. Include necessary imports, class definitions, and function placeholders (pass).\n"
            "3. Add detailed comments explaining the purpose of each block.\n"
            "Write the complete Python code to 'lite_output/3_mvp_skeleton.py'."
        ),
        expected_output="An executable Python file containing the complete architectural skeleton.",
        agent=architect
    )

    # 5. Assemble and Execute the Crew
    crew = Crew(
        agents=[analyst, resource_hunter, architect],
        tasks=[task_analysis, task_resources, task_coding],
        process=Process.sequential
    )

    result = crew.kickoff()
    
    print("\n\n########################")
    print("## Workflow Complete! Output files located in 'lite_output/' ##")
    print("########################\n")
    print(result)

if __name__ == "__main__":
    run_lite_crew()