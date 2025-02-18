from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class ChatHandler:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-pro")
        self.output_parser = StrOutputParser()
        self.conversation_history = []
        self.prompt_template = """
        You are a personal assistant, your job is to help the user with anything they ask.
        Keep your answers short and to the point. As if you are speaking to a friend.
        Do not answer questions with code responses or error messages. Avoid using technical jargon.
        Answer as if you are talking a human being in a conversation.
        Your response should make the user happy, the user's current emotion is {emotion}.

        Conversation so far:
        {history}

        Now, the user just said: 
        {user_prompt}

        Assistant:
        """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

    def format_history(self):
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.conversation_history])

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_input, emotion):
        self.add_message("user", user_input)
        chain = self.prompt | self.model | self.output_parser
        response = chain.invoke({
            "emotion": emotion,
            "user_prompt": user_input,
            "history": self.format_history()
        })
        self.add_message("assistant", response)
        return response
