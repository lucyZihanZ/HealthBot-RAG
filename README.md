# Healthbot-RAG: A Retrieval-Augmented Generation System for Medical Diagnostics

## Project Overview

This project introduces **healthbot-RAG**, a sophisticated Retrieval-Augmented Generation (RAG) pipeline meticulously engineered to deliver precise medical diagnoses and recommendations. Leveraging patient casefile briefs and an extensive corpus of medical guidelines, the system intelligently retrieves highly relevant information from a curated knowledge base. This retrieved context is then seamlessly integrated with a large language model (LLM) to synthesize coherent, accurate, and contextually informed medical diagnoses. This system represents a robust solution for enhancing diagnostic efficiency and reliability.

## Core Architectural Components

The `healthbot-RAG` system is systematically structured into three interdependent modules, each contributing a critical function to the overall RAG workflow:

### 1. Data Preparation (`prepare.py`)

This foundational module is responsible for the meticulous construction and preprocessing of the knowledge base.

* **Guideline Ingestion**: Programmatically reads and stores the comprehensive medical guidelines from `guidelines.txt` into an accessible format.

* **Intelligent Text Chunking**: Implements a strategic approach to segment the extensive medical guideline document into smaller, semantically coherent text chunks. This process is crucial for mitigating information loss during embedding and optimizing the granularity of subsequent retrieval operations. Custom, native Python logic is employed for this task to ensure environmental compatibility.

* **High-Fidelity Embedding Generation**: Generates dense vector embeddings for each preprocessed text chunk utilizing a carefully selected Sentence Transformer model. This transformation converts textual data into a high-dimensional numerical representation, enabling efficient and accurate semantic similarity searches. The choice of embedding model is a critical factor for system performance.

* **Robust Vector Database Management**: Facilitates the efficient loading, indexing, and persistence of these embeddings along with their corresponding text chunks, ensuring rapid retrieval capabilities.

### 2. Document Retrieval (`retrieve.py`)

This module is dedicated to the precise identification and extraction of the most pertinent information relevant to a given patient query.

* **Vector Database Initialization**: Initializes and configures the vector database by integrating the pre-generated text embeddings.

* **Advanced Similarity Search**: Transforms an incoming patient query into its vector embedding and executes a high-performance similarity search against the initialized vector database. This operation identifies and returns the top 'N' most semantically similar medical guideline chunks. The output is structured as a list of tuples, each comprising the calculated similarity score and the corresponding text chunk.

### 3. Response Generation (`generate.py`)

This module serves as the critical interface with the Large Language Model, orchestrating the synthesis of the final medical diagnosis.

* **LLM Integration**: Establishes a secure and efficient connection with the Google Gemini API, leveraging its advanced natural language generation capabilities.

* **Strategic Prompt Engineering**: Constructs a meticulously designed prompt that intelligently incorporates both the patient query and the most relevant retrieved medical guideline chunks. This structured prompt guides the LLM to generate an accurate, detailed, and appropriately formatted medical diagnosis. The LLM is explicitly instructed to render its response as a JSON object, ensuring parseability and consistency.

* **Coordinated Query Processing**: Manages the end-to-end process, from the initial retrieval of relevant documents to the subsequent generation of the diagnosis via the integrated LLM. An optional `optimize_query` function is provided for potential prompt refinement.

## Key Features

* **Modular RAG Pipeline**: A clear, segmented architecture for easy maintainability and scalability.

* **Customizable Text Chunking**: Flexibility in defining chunking strategies to optimize information granularity.

* **Configurable Embedding Models**: Allows for experimentation with various Sentence Transformer models to achieve optimal semantic representation.

* **Dynamic Document Retrieval**: Efficiently identifies and ranks relevant medical guidelines based on query similarity.

* **Structured LLM Output**: Ensures consistent, machine-readable JSON responses from the Gemini API for straightforward integration and parsing.

* **Comprehensive Evaluation Framework**: Built-in metrics for assessing response quality and document relevancy.

## Technology Stack

* **Python**: The primary programming language for all modules.

* **Google Gemini API**: Utilized for its advanced large language model capabilities and robust content generation.

* **Sentence Transformers**: An integral library for generating high-quality, semantically rich text embeddings.

* **JSON**: Employed for standardized data interchange and structured output from the LLM.

## Project Structure

The project's functional components are meticulously organized within the following file hierarchy:

* `prepare.py`: Encapsulates all data preparation logic, including guideline ingestion, text chunking, and embedding generation.

* `retrieve.py`: Manages the vector database operations and executes the relevant document retrieval process.

* `generate.py`: Orchestrates the interaction with the Gemini API, prompt construction, and final medical diagnosis generation.

* `medical_generalization.csv`: The dataset containing patient queries (`prompt`) and ground truth answers (`answer`), serving as a vital resource for system evaluation.

* `guidelines.txt`: The authoritative source document containing the comprehensive medical guidelines.

* `run_tests.py`: A suite of unit tests designed to verify the functional correctness and validate the return types of the implemented components. *(Note: This file is for developer convenience and is not part of the final submission.)*

* `engine.py`: A dedicated evaluation script for assessing the `healthbot-RAG` system's performance against a sample dataset. *(Note: This file is for internal evaluation and is not part of the final submission.)*

## Evaluation Methodology

The performance of the `healthbot-RAG` system is rigorously evaluated against two critical performance indicators:

1. **Quality of Response**: This metric quantifies the accuracy and clinical soundness of the generated medical diagnoses when compared against established correct answers.

2. **Document Relevancy (Mean Reciprocal Rank - MRR)**: This metric assesses the effectiveness of the retrieval mechanism by measuring how highly ranked the truly relevant medical guideline chunks are within the set of retrieved documents. A superior MRR score indicates that the most critical information is efficiently located and presented earlier in the retrieval results.

The ultimate system performance is synthesized as an average of these two metrics, providing a holistic assessment of the RAG pipeline's efficacy.

## Getting Started

To establish and operate this project environment, please follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd healthbot-RAG
   ```
2. **Install Dependencies**:
   ```bash
   transformers
   torch
   pandas
   google-generativeai
   ```
3. **Usage Example**:
   ```python
   from generate import Generator

   generator = Generator(api_key="YOUR_GEMINI_API_KEY")

   patient_query = "A 60-year-old male presents with persistent cough, shortness of breath, and fatigue for the past month."
   diagnosis_result = generator.process_query(patient_query)

   print(diagnosis_result)
   ```

4.**Future Improvements**

- Implement adaptive or context-aware chunking  
- Fine-tune embedding models for improved retrieval  
- Integrate advanced prompt optimization techniques  
- Add feedback loops or RLHF for model refinement  
- Explore Dockerization and scalable deployment strategies


   
   
