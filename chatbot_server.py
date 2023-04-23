import gradio as gr


class ChatbotGradio:
    def __init__(self, bot) -> None:
        self.bot = bot

        self.new_app()

    def submit_message(self, user_message, history):
        return (
            "",
            history + [[user_message, None]],
        )

    def respond_and_update_history(self, bot, history):
        user_message = history[-1][0]

        response = bot.generate_response(user_message)

        history[-1][1] = response
        return history

    def clear(self, state):
        state.memory.clear()

        return (
            "",
            None,
        )

    def new_app(self):
        with gr.Blocks() as demo:
            state = gr.State(self.bot)
            chatbot = gr.Chatbot(label=f"{self.bot.model_name}")

            msg = gr.Textbox(label="Input")
            clear = gr.Button("Clear")

            msg.submit(
                self.submit_message,
                [msg, chatbot],
                [msg, chatbot],
                queue=False,
            ).then(
                self.respond_and_update_history,
                [state, chatbot],
                [chatbot],
            )
            clear.click(
                self.clear,
                [state],
                [msg, chatbot],
            )

        self.demo = demo

    def run(self):
        self.demo.launch()
