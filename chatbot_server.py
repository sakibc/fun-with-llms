import gradio as gr


class ChatbotServer:
    def __init__(self, bot) -> None:
        self.bot = bot

        self.new_app()

    def submit_message(self, session_state, user_message, history):
        self.bot.submit_message(session_state, user_message)

        return (
            "",
            history + [[user_message, None]],
            self.bot.raw_dialogue(session_state),
        )

    def update_dialogue(self, session_state):
        return self.bot.raw_dialogue(session_state)

    def respond_and_update_history(self, session_state, history):
        self.bot.generate_response(session_state)
        bot_message = self.bot.get_response(session_state)

        history[-1][1] = bot_message
        return history

    def clear(self, instruction, user_label, chatbot_label):
        return (
            self.bot.new_session_state_non_default(
                instruction, user_label, chatbot_label
            ),
            None,
            "",
        )

    def update_instruction(self, conversation, instruction):
        self.bot.set_instruction(conversation, instruction)

    def update_user_label(self, conversation, user_label):
        self.bot.set_user_label(conversation, user_label)

    def update_chatbot_label(self, conversation, chatbot_label):
        self.bot.set_chatbot_label(conversation, chatbot_label)

    def new_app(self):
        with gr.Blocks() as demo:
            # create conversation session state
            session_state = gr.State(self.bot.new_session_state())

            # ui elements
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        label=f"{self.bot.model_name} as {self.bot.chatbot_name}"
                    )
                    msg = gr.Textbox(label="Input")
                    clear = gr.Button("Clear")
                with gr.Column():
                    instruction = gr.Textbox(
                        label="Instruction", value=self.bot.get_instruction
                    )

                    with gr.Row():
                        user_label = gr.Textbox(
                            label="User label", value=self.bot.get_user_label
                        )
                        chatbot_label = gr.Textbox(
                            label="Bot label", value=self.bot.get_chatbot_label
                        )

                    raw_dialogue = gr.Textbox(label="Dialogue")

            # event listeners
            instruction.change(self.update_instruction, [session_state, instruction])
            user_label.change(self.update_user_label, [session_state, user_label])
            chatbot_label.change(
                self.update_chatbot_label, [session_state, chatbot_label]
            )

            msg.submit(
                self.submit_message,
                [session_state, msg, chatbot],
                [msg, chatbot, raw_dialogue],
                queue=False,
            ).then(
                self.respond_and_update_history,
                [session_state, chatbot],
                [chatbot],
            ).then(
                self.update_dialogue,
                [session_state],
                [raw_dialogue],
            )
            clear.click(
                self.clear,
                [instruction, user_label, chatbot_label],
                [session_state, chatbot, raw_dialogue],
            )

        self.demo = demo

    def run(self):
        self.demo.launch()
