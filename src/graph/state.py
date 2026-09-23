from typing import TypedDict, List, Optional



class AgentState(TypedDict, total=False):
    asset_id: str
    thread_id: str
    query: str
    intent: str
    doc_results: List[dict]
    api_results: dict
    summary: str
    report_path: Optional[str]
    email_status: Optional[str]
    answer: str
    
    def __init__(self):
        pass
    
    