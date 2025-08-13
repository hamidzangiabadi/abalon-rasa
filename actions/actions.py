from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from llama_cpp import Llama
import warnings

warnings.filterwarnings("ignore")

print("🔄 Loading LLM model...")

llm = Llama(
    model_path="./maral-model/Maral-7B-alpha-1-Q4_K_M.gguf",
    n_ctx=1024,      # Back to reasonable size
    n_batch=128,     # Back to working size
    n_threads=6,
    n_gpu_layers=0,
    f16_kv=True,     # Back on for stability
    use_mlock=True,
    verbose=False
)

class ActionLocalLLMResponse(Action):
    def name(self):
        return "action_local_llm_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):

        user_input = tracker.latest_message.get('text', '').strip()
        print(f"📥 User: {user_input}")

        # Simple but effective prompt
        prompt = f"""شما کارشناس پشتیبانی آبالون کلود هستید. پاسخ کوتاه و مفید بدهید.

کاربر: {user_input}
پشتیبانی آبالون:"""

        try:
            response = llm(
                prompt, 
                max_tokens=100,
                temperature=0.3,
                stop=["کاربر:", "پشتیبانی آبالون:"],
                echo=False
            )
            
            reply = response["choices"][0]["text"].strip()
            
            # Simple validation
            if not reply or len(reply) < 5:
                reply = "سلام! من پشتیبان آبالون کلود هستم. چطور می‌تونم کمکتون کنم؟"

            dispatcher.utter_message(reply)
            print(f"✅ Bot: {reply}")

        except Exception as e:
            print(f"❌ Error: {e}")
            dispatcher.utter_message("سلام! من پشتیبان آبالون کلود هستم. چطور می‌تونم کمکتون کنم؟")

        return []
