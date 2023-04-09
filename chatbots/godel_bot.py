from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class Conversation:
    def __init__(self, instruction, knowledge) -> None:
        self.instruction = instruction
        self.knowledge = knowledge
        self.dialogue = []


class GodelBot:
    model = None
    tokenizer = None

    def __init__(self) -> None:
        model_name = "microsoft/GODEL-v1_1-large-seq2seq"
        print(f"Using {model_name}")

        self.load_tokenizer(model_name)
        self.load_model(model_name)

    def load_tokenizer(self, model_name):
        if GodelBot.tokenizer is None:
            print("Loading tokenizer...")

            GodelBot.tokenizer = AutoTokenizer.from_pretrained(model_name)

            print("Tokenizer loaded.")

    def load_model(self, model_name):
        if GodelBot.model is None:
            print("Loading model...")

            GodelBot.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
                "cuda"
            )

            print("Model loaded.")

    def generate_response(self, conversation):
        knowledge = ""

        if conversation.knowledge != "":
            knowledge = "[KNOWLEDGE] " + conversation.knowledge

        dialogue = " EOS ".join(conversation.dialogue)
        query = f"{conversation.instruction} [CONTEXT] {dialogue} {knowledge}"

        input_ids = self.tokenizer(f"{query}", return_tensors="pt").to("cuda").input_ids
        outputs = self.model.generate(
            input_ids,
            max_length=512,
            min_length=8,
            temperature=0.7,
            do_sample=True,
            top_k=0,
            top_p=0.9,
        )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        conversation.dialogue.append(output)

    def submit_message(self, conversation, user_message):
        conversation.dialogue.append(user_message)

    def get_response(self, conversation):
        return conversation.dialogue[-1]

    def new_session_state(self):
        return Conversation(
            instruction="Instruction: given a dialogue context, you need to provide helpful and informative responses.",
            knowledge="""I am a helpful robot. I know kung fu and how to make an awesome-tasting milkshake. The secret is to add crushed almonds.
Nest (NestJS) is a framework for building efficient, scalable Node.js server-side applications. It uses progressive JavaScript, is built with and fully supports TypeScript (yet still enables developers to code in pure JavaScript) and combines elements of OOP (Object Oriented Programming), FP (Functional Programming), and FRP (Functional Reactive Programming).
Under the hood, Nest makes use of robust HTTP Server frameworks like Express (the default) and optionally can be configured to use Fastify as well!
Nest provides a level of abstraction above these common Node.js frameworks (Express/Fastify), but also exposes their APIs directly to the developer. This gives developers the freedom to use the myriad of third-party modules which are available for the underlying platform.
""",
        )
