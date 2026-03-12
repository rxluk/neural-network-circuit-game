from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

intents = {
    "appointment": [
        "quero marcar horário",
        "preciso cortar o cabelo",
        "tem horário disponível",
        "quero um corte"
    ],
    "price": [
        "quanto custa",
        "qual o preço",
        "valor do corte"
    ],
    "location": [
        "onde fica",
        "qual o endereço",
        "onde vocês estão"
    ]
}

# gerar embeddings das frases de intenção
intent_vectors = {}
for intent, phrases in intents.items():
    intent_vectors[intent] = model.encode(phrases)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_intent(text):
    emb = model.encode(text)

    best_intent = None
    best_score = -1

    for intent, vectors in intent_vectors.items():
        for v in vectors:
            score = cosine(emb, v)
            if score > best_score:
                best_score = score
                best_intent = intent

    return best_intent, best_score

while True:
    text = input("Você: ")
    intent, score = detect_intent(text)
    print("Intent:", intent, "score:", round(score, 3))