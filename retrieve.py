from typing import List, Tuple
from prepare import DocumentVectorizer

class DocumentRetriever:
    def __init__(self, limit=1):
        self.dv = DocumentVectorizer()
        self.limit = limit
        self.embeddings, self.text_chunks = self.dv.load_database()
        
        

    def retrieve(self, query) -> List[Tuple[float, str]]:
        '''
        This function retrieves the most relevant documents to the query from the guidelines.txt file.
        It uses the embeddings to search for the most relevant documents.
        
        It retrieves the top (self.limit) documents from the guidelines.txt file. The default limit is 1, 
        so it will return the most relevant document to the query (with respect to the embedding space.)
        '''
        # query = self.optimize_query(query)
        text_chunks = self.text_chunks # TODO: store the guidelines.txt as a string to this variable
        results = self.embeddings.search(query, limit = self.limit)
        return [(score, text_chunks[uid]) for uid, score in results]