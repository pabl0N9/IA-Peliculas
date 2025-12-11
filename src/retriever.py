from src import utils

class Retriever:
    def __init__(self, docs):
        texts = [d["text"] for d in docs]
        metas = [{"text": d["text"], "meta": d["meta"]} for d in docs]
        self.engine = utils.TfidfRetriever(texts, metas)

    def search(self, query, k=5):
        return self.engine.search(query, k=k)
