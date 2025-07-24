# Fusion RAG: A Streamlit-Powered PDF Q&A System with Fusion-in-Decoder

[![GitHub Stars](https://img.shields.io/github/stars/armanjscript/Fusion-RAG?style=social)](https://github.com/armanjscript/Fusion-RAG)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Description

Welcome to **Fusion RAG**, a powerful web-based application designed to answer questions based on the content of uploaded PDF documents. This project leverages the **Fusion-in-Decoder (FiD)** approach for **Retrieval-Augmented Generation (RAG)**, combining semantic similarity, technical term relevance, and recency to deliver accurate and contextually relevant responses. Built with modern technologies like **Streamlit**, **LangChain**, **Chroma**, and **Ollama**, Fusion RAG is ideal for researchers, students, or professionals who need to extract insights from documents efficiently.

The application features a user-friendly interface where you can upload PDFs, ask questions in a conversational format, and receive detailed, real-time responses with citations to the source documents for transparency and verifiability.

## Features

| Feature | Description |
|---------|-------------|
| **PDF Upload & Processing** | Upload multiple PDF files, which are automatically split into chunks and indexed for querying. |
| **Advanced Retrieval** | Uses a custom FiDRetriever that combines semantic similarity (via Chroma), technical term relevance (via TF-IDF), and recency to fetch the most relevant document chunks. |
| **Conversational Interface** | Ask questions in a chat-like interface and receive detailed answers based on document content. |
| **Real-Time Responses** | Answers are streamed in real-time for a seamless user experience. |
| **Error Handling** | Robust error handling and cleanup mechanisms ensure reliable operation. |
| **Source Citations** | Responses include references to the source documents, enhancing trust and verifiability. |

## How It Works

The Fusion RAG system employs a sophisticated pipeline to process documents and generate answers. Here’s a detailed breakdown of the process:

### Document Processing
- **PDF Loading**: Uploaded PDFs are loaded using `PyPDFLoader` from LangChain.
- **Text Splitting**: Documents are split into manageable chunks (1000 characters, 200-character overlap) using `RecursiveCharacterTextSplitter`.
- **Metadata Tagging**: Each chunk is tagged with metadata, such as the source file name, for traceability.

### Retrieval
- **Vector Search**: Documents are embedded into a high-dimensional vector space using `OllamaEmbeddings` (model: `nomic-embed-text:latest`) and stored in a Chroma vector store. This enables semantic similarity searches.
- **FiDRetriever**: A custom retriever that scores documents based on:
  - **Semantic Similarity**: Using vector embeddings to find contextually relevant chunks.
  - **Technical Term Relevance**: Using TF-IDF to prioritize chunks with important keywords.
  - **Recency**: Prioritizing recently seen documents for relevance.
- The top 2 most relevant documents are retrieved for each query.

### Generation
- **FiDChain**: Combines the retriever with an `OllamaLLM` (model: `qwen2.5:latest`, temperature: 0.3) to generate a detailed response.
- The response is structured to include an analysis summary, recommended approach, reference standards, and citations to the source documents.

### Diagram of the RAG Pipeline
```mermaid
graph LR
    A[User Query] --> B[FiDRetriever]
    B --> C[Vector Search (Chroma)]
    C --> D[Embeddings (OllamaEmbeddings)]
    B --> E[TF-IDF Scoring]
    B --> F[Recency Scoring]
    C --> G[Retrieved Documents]
    E --> G
    F --> G
    G --> H[Formatter]
    H --> I[Prompt Template]
    I --> J[LLM]
    J --> K[Output Parser]
    K --> L[Response]
```

This diagram can be rendered in GitHub to visualize the pipeline from query to response.

## Environment Setup

To run Fusion RAG, you’ll need to set up the following:

- **Python 3.8 or later**: Ensure Python is installed on your system. Download from [python.org](https://www.python.org/downloads/).
- **Ollama**: A tool for running large language models locally. Install it based on your operating system:
  - **Windows**: Download the installer from [Ollama Download](https://ollama.com/download) and run it.
  - **macOS**: Download the installer from [Ollama Download](https://ollama.com/download), unzip it, and drag the `Ollama.app` to your Applications folder.
  - **Linux**: Run the installation script as per the [Ollama GitHub repository](https://github.com/ollama/ollama).
- **Python Libraries**: Install the required dependencies listed in `requirements.txt`.

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/armanjscript/Fusion-RAG.git
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd Fusion-RAG
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` file includes dependencies like `streamlit`, `langchain`, `chromadb`, `langchain-ollama`, `scikit-learn`, and others.

4. **Start Ollama**:
   Ensure the Ollama service is running on your system. Follow the instructions from the [Ollama GitHub repository](https://github.com/ollama/ollama) to start the service.

5. **Run the Streamlit App**:
   ```bash
   streamlit run fusion_rag.py
   ```
   This will launch the app in your default web browser.

## Usage

1. **Upload PDFs**:
   - In the sidebar, use the file uploader to select one or more PDF files.
   - The files are saved locally in the `uploaded_pdfs` directory and indexed in a Chroma database (`chroma_db`).

2. **Ask Questions**:
   - Enter your query in the chat input field in the main interface.
   - The chatbot will retrieve relevant document chunks, generate a response, and stream it in real-time.
   - Responses include citations to the source documents for reference.

3. **Clear Documents**:
   - Use the "Clear All Documents" button in the sidebar to reset the application by deleting uploaded files and the vector store.

## Configuration

The system uses fixed parameters for optimal performance:
- **Temperature**: Set to 0.3 for controlled randomness in the language model’s responses.
- **Retrieval**: Retrieves the top 2 documents based on a combination of semantic similarity, TF-IDF scoring, and recency.

To modify these parameters, you can edit the `fusion_rag.py` file to adjust the `OllamaLLM` temperature or the `FiDRetriever` scoring logic.

## Contributing

We welcome contributions to enhance Fusion RAG! To contribute:
- Fork the repository.
- Make your changes in a new branch.
- Submit a pull request with a clear description of your changes.

For detailed guidelines, refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, feedback, or collaboration opportunities, please reach out:
- **Email**: [armannew73@gmail.com]
- **GitHub Issues**: Open an issue on this repository for bug reports or feature requests.

## Acknowledgments

This project builds on the following open-source technologies:
- [Streamlit](https://streamlit.io/) for the web interface
- [LangChain](https://www.langchain.com/) for document processing and RAG pipeline
- [Chroma](https://www.trychroma.com/) for vector storage
- [Ollama](https://ollama.com/) for local language models and embeddings
- [Scikit-learn](https://scikit-learn.org/) for TF-IDF vectorization

## Citations
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://streamlit.io/)
- [Chroma Documentation](https://www.trychroma.com/)
- [Ollama Documentation](https://ollama.com/)

Thank you for exploring Fusion RAG! We hope it simplifies your document analysis tasks.

#AI #RAG #FiD #PDFQ&A #Streamlit #LangChain #Chroma #Ollama #MachineLearning #NaturalLanguageProcessing