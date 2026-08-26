# CloudDesk Customer Support AI

CloudDesk Customer Support AI is a Tier-1 customer support assistant for a fictional SaaS platform.

The system accepts a customer's issue in natural language, classifies the issue into a support category, checks the classification confidence, retrieves relevant FAQs using semantic search, and generates a grounded response using Gemini.

If the system is not confident enough or cannot find relevant knowledge-base information, it escalates the request to human support instead of guessing.

## Features

- Natural-language customer support chat
- Ticket classification using TF-IDF and Logistic Regression
- Three support categories:
  - Billing
  - Technical
  - Account Access
- Confidence-based escalation
- Semantic FAQ retrieval using Sentence Transformers
- FAISS vector search
- Gemini-powered grounded responses
- Out-of-scope query detection
- Graceful LLM failure handling
- FastAPI backend
- Browser-based chat interface
- Automated test suite with pytest
- Environment-based configuration
- No hardcoded API keys

## System Architecture

```text
Customer
   |
   v
Frontend Chat UI
   |
   | POST /chat
   v
FastAPI Backend
   |
   v
Ticket Classifier
(TF-IDF + Logistic Regression)
   |
   v
Confidence Check
   |
   +----------------------+
   |                      |
Low confidence         Confident
   |                      |
   v                      v
Human Escalation       FAISS RAG Retrieval
                          |
                          +------------------+
                          |                  |
                    No relevant FAQ     Relevant FAQs
                          |                  |
                          v                  v
                    Human Escalation    Gemini LLM
                                             |
                                             v
                                      Grounded Response
                                             |
                                             v
                                        Customer