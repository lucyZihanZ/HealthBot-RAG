import pandas as pd
import json

from generate import Generator
from retrieve import DocumentRetriever

if __name__ == "__main__":
    question_df = pd.read_csv("medical_generalization.csv", index_col=0)
    question_df = question_df.iloc[:10,:] # just check the first 10 questions to save time.  you may change this number to check more/less questions.
    
    dr = DocumentRetriever()
    g = Generator()
    responses = []
    documents = []
    for i, row in question_df.iterrows():
        question = row["prompt"]
        docs = dr.retrieve(question)
        print("*****docs******",docs)
        documents.append(docs)
        response = g.process_query(question, docs)
        responses.append(response)
        
    for response, d, question, answer in zip(responses, documents, question_df['prompt'], question_df['answer']):
        # print("Question: ", question)
        # print("Your Answer: ", response)
        # print("Correct Answer: ", answer)
        # print("Documents: ", d)
        print("--------------------------------")