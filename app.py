import streamlit as st
import torch
import torch.nn as nn

# ---------------- MODEL CLASS ----------------
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


# ---------------- LOAD VOCAB ----------------
vocab = torch.load("vocab.pth")

# Ensure UNK token exists
if "<UNK>" not in vocab:
    vocab["<UNK>"] = 0

# ---------------- LOAD MODEL ----------------
vocab_size = len(vocab)
embedding_dim = 50   # SAME as training
hidden_dim = 64      # SAME as training
output_dim = len(vocab)

model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("qa_model.pth", map_location="cpu"))
model.eval()


# ---------------- HELPER FUNCTIONS ----------------
def text_to_indices(text, vocab):
    tokens = text.lower().split()
    return [vocab.get(word, vocab["<UNK>"]) for word in tokens]

def predict(model, question_text, vocab):
    indices = text_to_indices(question_text, vocab)
    tensor = torch.tensor(indices).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    predicted_idx = torch.argmax(output).item()
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return idx_to_word.get(predicted_idx, "<UNK>")


# ---------------- STREAMLIT UI ----------------
st.title("🧠 Question Answering System")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question")
    else:
        answer = predict(model, question, vocab)
        st.success(f"Answer: {answer}")
