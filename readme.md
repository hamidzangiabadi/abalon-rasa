Run rasa:
rasa run --enable-api --cors "\*"

Run rasa actions:
rasa run actions

Run Ollama:
Ollama serve

path=~/Desktop/AbalonChatbot/rasacourse/first_project$

transformers:
https://huggingface.co/docs/hub/transformers

---

Front End: A chatbot interface that sends user questions to an API.

Load Balancer: Distributes requests across multiple chatbot instances.

Chatbot Instances (VPS/Cloud Instances): Multiple servers running a containerized version of your RAG pipeline. Each instance runs a quantized version of the Maral model (or another suitable LLM) on a strong CPU with ample RAM.

Vector Database: A centralized database (like Qdrant) that all chatbot instances use to retrieve relevant document chunks.

Orchestration Platform: A service like Kubernetes that manages the scaling of your chatbot instances up and down based on user traffic.

---
