import os
from app.rag import create_index, load_index_and_docs
from app.model_loader import load_phi_model
from app.generator import generate_response_phi

if not (os.path.exists('index.faiss') and os.path.exists('docs.pkl')):
    create_index('data/data.txt')

embedding_model, index, docs = load_index_and_docs()
tokenizer, model = load_phi_model()

print("RogerBot est prêt. Tape 'exit' pour quitter.")
while True:
    question = input("Toi : ")
    if question.lower() in ['exit', 'quit']:
        break
    answer, _ = generate_response_phi(question, tokenizer, model, embedding_model, index, docs)
    print("RogerBot :", answer)
