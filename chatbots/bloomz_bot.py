from transformers import AutoModelForCausalLM, AutoTokenizer


class Conversation:
    def __init__(self, chatbot_name) -> None:
        self.preamble = f"This is a dialogue between User and {chatbot_name}. {chatbot_name} is helpful, friendly, and eager to please. {chatbot_name} provides long, meaningful answers to all of User's questions."
        self.dialogue = ""

        self.last_response = ""


class BloomzBot:
    model = None
    tokenizer = None

    def __init__(self) -> None:
        self.model_name = "bigscience/bloomz-7b1"
        print(f"Using {self.model_name}")
        self.chatbot_name = "Chatbot"

        self.load_tokenizer(self.model_name)
        self.load_model(self.model_name)

    def load_tokenizer(self, model_name):
        if BloomzBot.tokenizer is None:
            print("Loading tokenizer...")

            BloomzBot.tokenizer = AutoTokenizer.from_pretrained(model_name)

            print("Tokenizer loaded.")

    def load_model(self, model_name):
        if BloomzBot.model is None:
            print("Loading model...")

            BloomzBot.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", load_in_8bit=True
            )

            print("Model loaded.")

    def generate_response(self, conversation):
        conversation.dialogue += f"\n{self.chatbot_name}: "

        input = f"{conversation.preamble}\n{conversation.dialogue}"
        inputs = self.tokenizer.encode(input, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=512)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        conversation.last_response = output[len(input) :].split("User:")[0].strip()
        conversation.dialogue += conversation.last_response

    def submit_message(self, conversation, user_message):
        if conversation.dialogue != "":
            conversation.dialogue += "\n"

        conversation.dialogue += f"User: {user_message.strip()}"

    def get_response(self, conversation):
        return conversation.last_response

    def new_session_state(self):
        return Conversation(self.chatbot_name)

    def raw_dialogue(self, conversation):
        return conversation.dialogue
