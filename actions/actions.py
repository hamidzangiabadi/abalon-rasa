from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from llama_cpp import Llama
# rag

print("🔄 Loading LLM model...")
llm = Llama(
    model_path="./maral-model/Maral-7B-alpha-1-Q4_K_M.gguf",
    n_ctx=256,
    n_gpu_layers=1,
    verbose=True
)

class ActionLocalLLMResponse(Action):
    def name(self):
        return "action_local_llm_response"

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get('text')
        print(f"📥 Received user input: {user_input}")

        prompt = f"User: {user_input}\nAssistant:"
        try:
            response = llm(prompt, max_tokens=1024, stop=["User:", "Assistant:"], echo=False)
            print("🧠 LLM raw response:", response)
            reply = response["choices"][0]["text"].strip()
            dispatcher.utter_message(reply)
            print("✅ Sent reply:", reply)
        except Exception as e:
            print("❌ Error in LLM response:", e)
            dispatcher.utter_message(text="مشکلی پیش آمده در پاسخ‌دهی مدل.")

        return []

