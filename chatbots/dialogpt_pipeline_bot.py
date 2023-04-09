from transformers import Conversation, pipeline


class DialoGPTPipelineBot:
    model = None

    def __init__(self) -> None:
        model_name = "microsoft/DialoGPT-large"
        print(f"Using {model_name} with ConversationalPipeline")
        self.load_model(model_name)

    def load_model(self, model_name):
        if DialoGPTPipelineBot.model is None:
            print("Loading pipeline...")

            DialoGPTPipelineBot.model = pipeline(
                "conversational", model=model_name, device="cuda:0"
            )

            print("Pipeline loaded.")

    def generate_response(self, conversation):
        return self.model(conversation)

    def submit_message(self, conversation, user_message):
        return conversation.add_user_input(user_message)

    def get_response(self, conversation):
        return conversation.generated_responses[-1]

    def new_session_state(self):
        return Conversation()
