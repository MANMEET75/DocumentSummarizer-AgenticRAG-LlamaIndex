# DocumentSummarizer-AgenticRAG-LlamaIndex

# HindBot
<img src="static/demo.gif">

## Agent Overview

This agent I have created uses the Llama Index framework, enabling it to summarize documents uploaded by users in PDF format. It has the ability to leverage Hugging Face, OpenAI, as well as Google Gemini large language models. It is built using FastAPI, allowing us to call it from anywhere we need when creating larger projects.


## Technical Stack
- **Modular Directory Structure:** Organized codebase for easy maintenance and scalability
- **Google Palm Embeddings:** Convert words to numerical representations for efficient processing
- **Docker:** Containerize the application for easy deployment and management
- **FastAPI:** Create a robust and interactive web interface with a RESTful API


## CI/CD Pipelines
- **GitHub Actions:** Automated CI/CD pipelines for continuous integration and deployment
- Automated testing and validation of code changes
- Automated deployment to AWS Cloud infrastructure

## Cloud Deployment
- **AWS Cloud:** Deployed on Amazon Web Services (AWS) for scalability and reliability
- **EC2 Instance:** Running the application on a secure and scalable EC2 instance
-** ECR:** Using Amazon Elastic Container Registry (ECR) for container image management

## Key Features
- End-to-end Agentic RAG implementation
- Modular directory structure for easy maintenance
- Google Palm embeddings for efficient word representation
- FastAPI-powered web interface for interactive user experience
- Automated CI/CD pipelines using GitHub Actions
- Scalable and secure deployment on AWS Cloud infrastructure

## Instructions for Execution
### 1. Cloning the Repository
```bash

git clone https://github.com/MANMEET75/HindBot.git
```
### 2. Creating the virtual environment using anaconda
```bash
conda create -p venv python=3.11 -y
```

### 3. Activate the virtual environment
```bash
conda activate venv/
```

### 4. Install the Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI application
```bash
uvicorn app:app --reload --port 8080
```

Enjoy Coding!
