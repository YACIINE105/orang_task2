from langchain_community.document_loaders import PyMuPDFLoader
from dataclasses import dataclass
from typing import List



@dataclass
class Document:
    page_content: str
    metadata : dict


class ProcessController:
    def __init__(self):
        pass
    
    def get_loader(self, file_path):
        mu = PyMuPDFLoader(file_path)
        loader = mu.load()
        if not loader:
            return None
        return loader
    
    def process_text(self, file_path):
        docs = self.get_loader(file_path=file_path)
        
        file_content = [content.page_content for content in docs]
        file_content_metadata = [content.metadata for content in docs]  
        chunks = self.process_simple_splitter(texts=file_content, metadatas=file_content_metadata)

        return chunks
    
    def process_simple_splitter(self, texts:List[str], metadatas:List[dict], chunk_size:int=300, splitter_tag:str="\n \n"):
        full_text = "".join(texts)
        
        lines = [doc.strip() for doc in full_text.split(splitter_tag) if len(doc.strip()) > 1] 

        chunks = []
        current_chunk = ""
        
        for line in lines:
            current_chunk += line + splitter_tag
            
            if len(current_chunk) >= chunk_size:
                
                chunks.append(Document(page_content=current_chunk.strip(), 
                                       metadata={}))
                current_chunk = ""
                
                
        if len(current_chunk) > 0 :
                        
                        chunks.append(Document(page_content=current_chunk.strip(), 
                                               metadata={}))
                        
        return chunks  




if __name__=='__main__':
    pro = ProcessController()
    file_path = "/home/yacine_105/orange_tasks/task2/NLP_0.5.pdf"
    chunks = pro.process_text(file_path=file_path)
    print(chunks)
    
    
    