import argparse
from src import utils
from src.retriever import Retriever
from src.chatbot import responder


def infer_intent(pregunta: str) -> str:
    q = pregunta.lower()
    if "album" in q:
        return "buscar_album"
    if "cancion" in q or "canción" in q or "song" in q:
        return "buscar_cancion"
    return "recomendar"


def main():
    parser = argparse.ArgumentParser(description="Chatbot musical institucional (CLI)")
    parser.add_argument("pregunta", nargs="*", help="Pregunta del usuario")
    args = parser.parse_args()
    pregunta = " ".join(args.pregunta) if args.pregunta else input("Pregunta: ")

    df = utils.load_catalog("data/catalogo.csv")
    politicas = utils.load_text("data/politicas.md")
    faq = utils.load_text("data/faq.md")
    docs = utils.build_docs(df)
    retriever = Retriever(docs)

    intent = infer_intent(pregunta)
    respuesta = responder(retriever, politicas, faq, pregunta, intent, llm_callable=None)
    print(respuesta)


if __name__ == "__main__":
    main()
