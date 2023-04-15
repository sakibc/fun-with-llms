class ChatbotCmd:
    def __init__(self, bot) -> None:
        self.bot = bot

        self.new_app()

    def submit_message(self, session_state, user_message):
        self.bot.submit_message(session_state, user_message)

    def new_app(self):
        self.session_state = self.bot.new_session_state()

    def run(self):
        print(
            f"\nRunning chatbot command line interface. You are now chatting with ${self.bot.model_name}."
        )
        print("Type 'quit' to exit the chatbot.\n")

        while True:
            user_message = input("You: ")
            if user_message == "quit":
                break

            self.bot.submit_message(self.session_state, user_message)

            self.bot.generate_response(self.session_state)
            print(
                f"\n{self.session_state.chatbot_name}: {self.session_state.last_response}\n"
            )
