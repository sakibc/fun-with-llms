class ChatbotCmd:
    def __init__(self, bot) -> None:
        self.bot = bot

    def run(self):
        print(
            f"\nRunning chatbot command line interface. You are now chatting with {self.bot.model_name}."
        )
        print("Type 'quit' to exit the chatbot.\n")

        while True:
            user_message = input("You: ")
            if user_message == "quit":
                break

            print("\nChatbot: ", end="")

            response = self.bot.generate_response(user_message)

            print(f"{response}\n")
