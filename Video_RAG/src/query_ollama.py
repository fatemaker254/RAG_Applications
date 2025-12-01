from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from config import CHROMA_DIR

llm = Ollama(model="phi3:mini")

db = Chroma(persist_directory=CHROMA_DIR)
retriever = db.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("\n🎤 Ask questions about the video (type exit to quit)\n")
while True:
    q = input(">> ")
    if q.lower() == "exit":
        break
    print("\n🎯 Answer:")
    print(qa.run(q), "\n")
