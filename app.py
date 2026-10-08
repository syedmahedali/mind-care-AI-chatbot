import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import re

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="MindCare AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #4a4a4a;
}

.subtitle {
    text-align: center;
    color: #666666;
    margin-bottom: 30px;
}

.user-message {
    background-color: black;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
}

.bot-message {
    background-color: lightgreen;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
}

.warning {
    background-color: #fff3cd;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    '<div class="title">🧠 MindCare AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">A simple AI chatbot for emotional support</div>',
    unsafe_allow_html=True
)

st.info(
    "MindCare AI provides general emotional support and is not a replacement "
    "for a mental-health professional."
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model_name = "microsoft/DialoGPT-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_ids" not in st.session_state:
    st.session_state.chat_ids = None

if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()


# ---------------------------------------------------
# BASIC OFFENSIVE WORD FILTER
# ---------------------------------------------------

OFFENSIVE_WORDS = [
    "idiot",
    "stupid",
    "dumb",
    "hate you",
    "shut up"
]


def contains_offensive_language(text):

    text = text.lower()

    for word in OFFENSIVE_WORDS:

        if re.search(r"\b" + re.escape(word) + r"\b", text):
            return True

    return False


# ---------------------------------------------------
# CRISIS DETECTION
# ---------------------------------------------------

CRISIS_KEYWORDS = [
    "kill myself",
    "suicide",
    "end my life",
    "want to die",
    "don't want to live",
    "dont want to live",
    "hurt myself",
    "harm myself",
    "self harm",
    "self-harm"
]


def detect_crisis(text):

    text = text.lower()

    for keyword in CRISIS_KEYWORDS:

        if keyword in text:
            return True

    return False


# ---------------------------------------------------
# EMPATHETIC RESPONSES
# ---------------------------------------------------

def empathetic_response(user_input):

    text = user_input.lower()

    if "sad" in text:
        return (
            "I'm sorry you're feeling sad. You don't have to handle "
            "everything alone. Would you like to tell me what's making "
            "you feel this way?"
        )

    if "lonely" in text:
        return (
            "Feeling lonely can be really difficult. I'm here to listen. "
            "If you're comfortable, you can tell me what's been happening."
        )

    if "stress" in text or "stressed" in text:
        return (
            "It sounds like you're dealing with a lot of stress. "
            "Taking a short break, breathing slowly, or talking to "
            "someone you trust may help."
        )

    if "anxiety" in text or "anxious" in text:
        return (
            "I'm sorry you're experiencing anxiety. Try taking a few "
            "slow breaths and focus on what you can control right now."
        )

    if "angry" in text or "anger" in text:
        return (
            "It's okay to feel angry. Taking some time away from the "
            "situation and expressing your feelings calmly may help."
        )

    if "depressed" in text or "depression" in text:
        return (
            "I'm sorry you're going through this. Talking with someone "
            "you trust or a qualified mental-health professional can be "
            "a helpful step."
        )

    return None


# ---------------------------------------------------
# GENERATE AI RESPONSE
# ---------------------------------------------------

def generate_response(user_input):

    # Crisis response
    if detect_crisis(user_input):

        return (
            "I'm really sorry you're going through such a difficult moment. "
            "Your safety is important. Please contact someone you trust and "
            "consider reaching out to a mental-health professional or local "
            "emergency/crisis service immediately. If you are in immediate "
            "danger, please contact your local emergency services."
        )

    # Offensive language
    if contains_offensive_language(user_input):

        return (
            "Let's keep the conversation respectful. I'm here to listen "
            "and support you."
        )

    # Predefined empathetic response
    empathy = empathetic_response(user_input)

    if empathy:
        return empathy

    # Prepare input
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Add previous conversation
    if st.session_state.chat_ids is not None:

        bot_input_ids = torch.cat(
            [
                st.session_state.chat_ids,
                new_input_ids
            ],
            dim=-1
        )

    else:

        bot_input_ids = new_input_ids

    # Limit conversation length
    bot_input_ids = bot_input_ids[:, -512:]

    # Generate response
    with torch.no_grad():

        chat_history_ids = model.generate(
            bot_input_ids,
            max_new_tokens=80,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=3
        )

    # Save conversation
    st.session_state.chat_ids = chat_history_ids

    # Decode only newly generated response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    response = response.strip()

    # Fallback
    if not response:

        response = (
            "I'm here to listen. Could you tell me a little more "
            "about what you're experiencing?"
        )

    return response


# ---------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------

for message in st.session_state.chat_history:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
            <b>You:</b> {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-message">
            <b>MindCare AI:</b> {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------

user_input = st.chat_input(
    "Tell me how you're feeling..."
)


# ---------------------------------------------------
# PROCESS MESSAGE
# ---------------------------------------------------

if user_input:

    user_input = user_input.strip()

    if user_input:

        # Add user message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # Generate response
        with st.spinner("MindCare AI is thinking..."):

            response = generate_response(user_input)

        # Add bot response
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # Refresh page
        st.rerun()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.title("🧠 MindCare AI")

    st.write(
        "A simple AI-based emotional support chatbot "
        "built using Python, Streamlit and Hugging Face Transformers."
    )

    st.divider()

    st.subheader("Features")

    st.write("✅ AI conversation")
    st.write("✅ Emotional support")
    st.write("✅ Basic offensive-language filter")
    st.write("✅ Crisis keyword detection")
    st.write("✅ Conversation history")
    st.write("✅ Session-based chat")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []
        st.session_state.chat_ids = None

        st.rerun()

    st.divider()

    st.caption(
        "⚠️ This chatbot is for educational purposes only "
        "and does not provide medical diagnosis or treatment."
    )