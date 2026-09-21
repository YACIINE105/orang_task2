import os
from datetime import datetime

def write_report(query: str, summary: str, out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"report_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Report\n\n**Request:** {query}\n\n## Summary\n\n{summary}\n")
    return path
