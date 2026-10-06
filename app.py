"""
Temple Guidance Chatbot

Flow:
User -> Chatbot -> Understand Question -> Search Temple Information
     -> Generate Answer -> Display Response -> Keep Chat History
"""

import os
import re

import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Temple Guidance Chatbot", page_icon="🛕", layout="wide")

TEMPLES = [
    "Sri Venkateswara Temple",
    "Sri Ramalingeswara Temple",
    "Sri Kanaka Durga Temple",
    "Sri Meenakshi Temple",
    "Sri Jagannath Temple",
]

# ---------------------------------------------------------------------
# Step 2 data: what kinds of questions the chatbot can understand
# ---------------------------------------------------------------------
INTENTS = {
    "timings": ["timing", "timings", "time", "open", "close", "opening", "closing",
                "hours", "darshan", "when", "morning", "evening"],
    "history": ["history", "built", "founded", "origin", "ancient", "story",
                "legend", "century", "old", "who"],
    "pooja": ["pooja", "puja", "seva", "abhishekam", "aarti", "archana",
              "ritual", "offering", "prasad", "prasadam"],
    "location": ["where", "location", "address", "reach", "route", "directions",
                 "bus", "train", "airport", "map", "near"],
    "rules": ["rule", "rules", "dress", "code", "allowed", "photography",
              "mobile", "footwear", "prohibited", "entry", "wear"],
}

STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "to", "what", "about", "tell", "me",
    "in", "at", "for", "and", "on", "it", "this", "that", "can", "i", "you",
    "do", "does", "how", "please", "temple", "there", "be",
}

# ---------------------------------------------------------------------
# Stored temple data (SAMPLE - replace with your real temple details)
# ---------------------------------------------------------------------
STORED_DATA = {
    "Sri Venkateswara Temple": {
        "timings": "The temple is open from 6:00 AM to 12:00 PM. (Sample, edit with real timings.)",
        "history": "This is an ancient temple dedicated to Lord Venkateswara and is visited by many devotees every year.",
        "pooja": "Daily poojas and sevas are performed. Please check the official temple website for the schedule and booking.",
        "location": "The temple is located in Tirumala, near Tirupati, Andhra Pradesh.",
        "rules": "Visitors should dress modestly and follow the temple rules on footwear, mobile phones and photography.",
    },
}

# ---------------------------------------------------------------------
# Step 2: Understand the question
# ---------------------------------------------------------------------
def detect_intent(question, last_intent=None):
    words = set(re.findall(r"\w+", question.lower()))
    best, best_score = "general", 0
    for intent, keywords in INTENTS.items():
        score = len(words & set(keywords))
        if score > best_score:
            best, best_score = intent, score
    # Short follow-up like "and in the evening?" -> reuse the previous topic
    if best == "general" and last_intent and last_intent != "general" and len(words) <= 6:
        return last_intent
    return best


def content_tokens(text):
    return {w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS}


# ---------------------------------------------------------------------
# Step 3: Search temple information (stored data + uploaded PDFs)
# ---------------------------------------------------------------------
def extract_text(file):
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text, size=800, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def get_knowledge(temple):
    items = []
    for intent, text in STORED_DATA.get(temple, {}).items():
        items.append({"text": text, "intent": intent, "source": "Stored temple data"})
    for chunk in st.session_state.pdf_chunks.get(temple, []):
        items.append({"text": chunk, "intent": None, "source": "Uploaded PDF"})
    return items


def search(question, intent, temple, history, k=4):
    query = content_tokens(question)
    # Very short follow-up: also use the previous user question for context
    if len(query) <= 2:
        prev_users = [m["content"] for m in history if m["role"] == "user"]
        if prev_users:
            query |= content_tokens(prev_users[-1])

    intent_words = set(INTENTS.get(intent, []))
    scored = []
    for item in get_knowledge(temple):
        text_tokens = content_tokens(item["text"])
        score = len(query & text_tokens)
        score += min(len(intent_words & text_tokens), 3)   # topic words
        if item["intent"] == intent and intent != "general":
            score += 5                                       # stored data for this topic
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


# ---------------------------------------------------------------------
# Step 4: Generate the answer
# ---------------------------------------------------------------------
def simple_answer(question, intent, results):
    """No API key: pick the best sentences from the found information."""
    wanted = content_tokens(question) | set(INTENTS.get(intent, []))
    sentences = []
    for item in results:
        sentences += re.split(r"(?<=[.!?])\s+", item["text"])
    scored = [(len(wanted & content_tokens(s)), i, s) for i, s in enumerate(sentences)]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored[:2], key=lambda x: x[1])
    if top:
        return " ".join(s for _, _, s in top)
    return results[0]["text"][:300]


def llm_answer(api_key, temple, question, results, history):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    context = "\n\n".join(r["text"] for r in results) or "No information found."
    system = (
        f"You are a helpful temple guidance assistant for {temple}. "
        "Answer in 1 to 3 simple sentences using only the provided temple information. "
        "If the information does not contain the answer, say you do not have it. "
        "Be polite and respectful."
    )
    past = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    messages = past + [
        {"role": "user", "content": f"Temple information:\n{context}\n\nQuestion: {question}"}
    ]
    response = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=500, system=system, messages=messages
    )
    return "".join(b.text for b in response.content if b.type == "text")


def generate_answer(api_key, temple, question, intent, results, history):
    if not results:
        return ("I could not find that information. Please upload a temple PDF "
                "and click **Add PDF**, or ask about timings, history, pooja, "
                "location or rules.")
    if api_key:
        try:
            return llm_answer(api_key, temple, question, results, history)
        except Exception as e:
            return f"Sorry, the AI request failed: {e}"
    return simple_answer(question, intent, results)


# ---------------------------------------------------------------------
# Step 6: Chat history (kept in session state)
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = {}
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

# No API key box: the chatbot answers directly from the temple information.
api_key = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🛕 Temple Settings")
    temple = st.selectbox("Select Temple", TEMPLES)
    st.info(f"Selected Temple:\n\n**{temple}**")

    st.subheader("📄 Temple Knowledge")
    pdf = st.file_uploader("Upload Temple PDF", type=["pdf"])

    if st.button("Add PDF"):
        if pdf is None:
            st.warning("Please upload a PDF first.")
        else:
            try:
                text = extract_text(pdf)
                if not text.strip():
                    st.error("No readable text found (the PDF may be scanned images).")
                else:
                    chunks = chunk_text(text)
                    st.session_state.pdf_chunks.setdefault(temple, []).extend(chunks)
                    st.success(f"Added {len(chunks)} sections to {temple}.")
            except Exception as e:
                st.error(f"Could not read PDF: {e}")

    st.caption(f"PDF sections loaded: {len(st.session_state.pdf_chunks.get(temple, []))}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_intent = None
        st.rerun()


# ---------------------------------------------------------------------
# Main chat screen
# ---------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
temple_pic = next(
    (os.path.join(APP_DIR, f) for f in ("temple.jpg", "temple.jpeg", "temple.png")
     if os.path.exists(os.path.join(APP_DIR, f))),
    None,
)

pic_col, title_col = st.columns([1, 5], vertical_alignment="center")
with pic_col:
    if temple_pic:
        st.image(temple_pic, width=110)
    else:
        st.markdown("# 🛕")   # shown if temple.jpg is not found
with title_col:
    st.title("Temple Guidance Chatbot")
    st.caption(f"Ask about {temple}: timings, history, pooja, location, rules.")

# Show the saved conversation
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("intent"):
            st.caption(f"Topic understood: {m['intent']}")

# Step 1: user asks a question
question = st.chat_input("Ask your question... (e.g. What are the temple timings?)")

if question:
    history = list(st.session_state.messages)          # earlier messages only

    with st.chat_message("user"):
        st.markdown(question)

    # Step 2: understand
    intent = detect_intent(question, st.session_state.last_intent)
    # Step 3: search
    results = search(question, intent, temple, history)
    # Step 4: generate
    answer = generate_answer(api_key, temple, question, intent, results, history)

    # Step 5: display
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"Topic understood: {intent}")
        if results:
            with st.expander("Sources used"):
                for r in results:
                    st.markdown(f"**{r['source']}**: {r['text'][:300]}")

    # Step 6: save history for follow-up questions
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "intent": intent}
    )
    st.session_state.last_intent = intent