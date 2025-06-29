from txtai.embeddings import Embeddings
import os
from typing import List, Tuple

class DocumentVectorizer:
    def __init__(self):
        self.document = open('guidelines.txt', 'r', encoding='utf-8').read() # TO DO: store the guidelines.txt as a string to this variable

    def chunk_text(self) -> List[str]:
        '''
        The text input is the string representation of the guidelines. 
        '''
        text = self.document
        chunks = [s.strip() for s in text.split('\. ') if s.strip()]
        return chunks
    def generate_embeddings(self, text_chunks: List[str]) -> Embeddings:
        """Create a txtai embeddings database and save it to disk.
        """
        os.makedirs('db', exist_ok=True)
        
        # what model should we use?
        embeddings = Embeddings(path="sentence-transformers/all-mpnet-base-v2") # TO DO: we should call the Embeddings class here
        # an index step is needed here too
        for i, chunk in enumerate(text_chunks):
            embeddings.index([(i, chunk, None)])
        
        embeddings.save(f"db/embeddings")
        return embeddings

    def load_database(self) -> Tuple[Embeddings, List[str]]:
        """Load the embeddings database and articles from disk.
        
        This function returns the embeddings and the corresponding text chunks.
        It loads the embeddings from your hard drive which is originally saved in the 
        generate_embeddings function.
        """
        embeddings = Embeddings(path="sentence-transformers/all-mpnet-base-v2") # we should call the Embeddings class here
        embeddings.load(f"db/embeddings")
        
        text_chunks = self.chunk_text()
        
        return embeddings, text_chunks