# Install Ollama on your window or you can download the desktop app

irm https://ollama.com/install.ps1 | iex

# Set Fireworks key for remote routing

export FIREWORKS_API_KEY="your-key"

# Run and Install the requirement

source venv/bin/activate
pip install -r my_routing_agent/requirements.txt

# Pull the local model (required for local routing)
ollama pull llama3.2

# Launch the chatbot
streamlit run app.py

# Check if it's up
curl http://localhost:11434/v1/models

# If connection refused, start the service:
sudo systemctl start ollama        # if installed as a systemd service

# Then run this command

ollama pull llama3.2:3b

# Verify the AI

ollama run llama3.2:3b "Hello"
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Say hi in one word."}]
  }'

# If it does response in your terminal then it work

# You have to activate your env environment first then run this command 

Source venv/bin/activate
PYTHONPATH=. python -m my_routing_agent.main "What is 17 * 23?"


# Other test cases

# Complex task → remote
PYTHONPATH=. python -m my_routing_agent.main "Analyze microservices vs monolith step by step"
# With image
PYTHONPATH=. python -m my_routing_agent.main "Describe this image" --image photo.jpg
# Force JSON schema output on remote
PYTHONPATH=. python -m my_routing_agent.main "Summarize this bug report" --json