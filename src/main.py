import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool
from dotenv import load_dotenv

# 載入 .env 中的 API KEY (需要 OPENAI_API_KEY 和 SERPER_API_KEY)
load_dotenv()

def run_lite_crew():
    print("=== 啟動 SaaS Launchpad Crew (Lite Version) ===")
    
    # 1. 獲取用戶輸入 (取代原本的 PDF 讀取)
    idea = input("\n請輸入你的專案點子 (例如: 一個幫我自動記帳的 AI Line Bot): ")
    if not idea:
        print("未輸入點子，程式結束。")
        return

    # 2. 初始化核心工具 (只保留搜索和寫檔)
    search_tool = SerperDevTool()
    file_writer = FileWriterTool()

    # 3. 定義 Agents (保留原本的三個角色，但 Backstory 精簡化)
    
    # 角色 A: 分析師
    analyst = Agent(
        role='首席產品分析師',
        goal='分析用戶點子，產出明確的功能規格書 (PRD)',
        backstory='你擅長將模糊的點子轉化為邏輯清晰的開發文檔，並能識別潛在的市場風險。',
        tools=[search_tool, file_writer],
        verbose=True,
        memory=True
    )

    # 角色 B: 資源獵人
    resource_hunter = Agent(
        role='技術資源探勘者',
        goal='尋找最適合的 Python 庫、API 和開源專案來實作需求',
        backstory='你熟悉 Python 生態系與 Github，總能找到現成的工具來避免重複造輪子。',
        tools=[search_tool, file_writer],
        verbose=True,
        memory=True
    )

    # 角色 C: 架構師
    architect = Agent(
        role='資深技術架構師',
        goal='根據分析與資源，產出 MVP 的核心程式碼結構',
        backstory='你擅長快速搭建可運行的 MVP，注重代碼的簡潔與模組化。',
        tools=[file_writer], # 架構師通常不需要上網，專注寫 Code
        verbose=True,
        memory=True
    )

    # 4. 定義任務 (大幅合併，變為 3 個核心任務)

    # 任務 1: 產品規格 (合併了原本的 Context, Objective, Feasibility)
    task_analysis = Task(
        description=(
            f"針對用戶的點子：'{idea}' 進行分析。\n"
            "1. 定義 3-5 個核心功能 (MVP scope)。\n"
            "2. 識別潛在的技術難點。\n"
            "3. 使用 search_tool 搜尋市面上的類似產品，列出 2 個競品。\n"
            "將結果寫入 'lite_output/1_spec.md'。"
        ),
        expected_output="一份包含功能列表、難點與競品的 Markdown 文件。",
        agent=analyst
    )

    # 任務 2: 技術堆疊 (合併了原本的 Resource, Discovery, Dataset)
    task_resources = Task(
        description=(
            "根據分析師的規格書，推薦技術堆疊。\n"
            "1. 搜尋並列出最適合的 3 個 Python Library (需包含 pip 安裝指令)。\n"
            "2. 如果需要外部 API (如 OpenAI, Line, Weather)，請列出推薦的服務商。\n"
            "將結果寫入 'lite_output/2_tech_stack.md'。"
        ),
        expected_output="一份包含 Python 套件清單與 API 推薦的 Markdown 文件。",
        agent=resource_hunter
    )

    # 任務 3: 核心代碼 (合併了原本的 Architecture, Template, Code)
    task_coding = Task(
        description=(
            "根據規格書與技術堆疊，撰寫一份核心的 'main.py'。\n"
            "1. 這不是要寫出完整的產品，而是 '骨架 (Skeleton)'。\n"
            "2. 包含必要的 imports, 類別定義 (Class) 和函數佔位符 (pass)。\n"
            "3. 在程式碼中加入詳細註解，說明每個區塊的作用。\n"
            "將完整的 Python 程式碼寫入 'lite_output/3_mvp_skeleton.py'。"
        ),
        expected_output="一個可執行的 Python 檔案，包含完整的架構骨架。",
        agent=architect
    )

    # 5. 組建與執行 Crew
    crew = Crew(
        agents=[analyst, resource_hunter, architect],
        tasks=[task_analysis, task_resources, task_coding],
        process=Process.sequential
    )

    result = crew.kickoff()
    
    print("\n\n########################")
    print("## 任務完成！輸出檔案位於 lite_output/ 目錄 ##")
    print("########################\n")
    print(result)

if __name__ == "__main__":
    run_lite_crew()