import langchain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import FileSystemLoader
from langchain.chains import ConversationalRetrievalChain
import pyttsx3

# Load dataset (replace path with your actual dataset location)
loader = FileSystemLoader("Tire_Recycling_Open_Source.pdf")
documents = loader.load()

# Create vector database
vectordb = Chroma(embedding_function=OpenAIEmbeddings())
vectordb.index(documents)

# Set up language model and chain
llm = OpenAI(temperature=0.5)  # Adjust temperature as needed
chain = ConversationalRetrievalChain.from_llm(
    llm, vectordb.as_retriever(), condense_question_prompt="Answer this question based on the tire recycling dataset:"
)

# Initialize text-to-speech engine
engine = pyttsx3.init()

def get_voice_input():
    text = engine.say("Ask a question about tire recycling:")
    engine.runAndWait()
    # Use a speech-to-text library or API to capture user's spoken question
    return captured_question

def provide_voice_output(answer):
    engine.say(answer)
    engine.runAndWait()

# Main interaction loop
while True:
    question = get_voice_input()
    response = chain({"question": question})
    provide_voice_output(response["answer"])
