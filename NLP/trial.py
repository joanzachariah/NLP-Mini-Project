# app.py - Simplified WhatsApp-like Hindi Predictive Typing
import streamlit as st
from transformers import pipeline
import re
import time

# ------------------------
# SETTINGS
# ------------------------
MODEL_NAME = "surajp/gpt2-hindi"
MAX_NEW_TOKENS = 12
N_SUGGESTIONS = 4
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 0.8

# ------------------------
# PAGE SETUP
# ------------------------
st.set_page_config(
    page_title="हिंदी प्रेडिक्टिव टाइपिंग", 
    page_icon="💬", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------
# CUSTOM CSS
# ------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        font-family: "Noto Sans Devanagari", Arial, sans-serif;
    }
    
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #25D366;
        margin: 20px 0;
        text-shadow: 0 0 20px rgba(37, 211, 102, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #8b949e;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    
    .suggestion-container {
        background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #30363d;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    
    .suggestion-title {
        color: #25D366;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .stats-container {
        background: #161b22;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #25D366;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Custom input styling */
    .stTextInput > div > div > input {
        background: #21262d !important;
        border: 2px solid #25D366 !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 18px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 15px rgba(37, 211, 102, 0.1) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #25D366 !important;
        box-shadow: 0 0 0 3px rgba(37, 211, 102, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(37, 211, 102, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 211, 102, 0.4) !important;
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# TITLE
# ------------------------
st.markdown('<h1 class="main-title">💬 हिंदी प्रेडिक्टिव टाइपिंग</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">WhatsApp जैसा स्मार्ट सुझाव सिस्टम</p>', unsafe_allow_html=True)

# ------------------------
# MODEL LOADING
# ------------------------
@st.cache_resource
def load_hindi_model():
    """Load the Hindi GPT-2 model"""
    try:
        with st.spinner("🤖 हिंदी मॉडल लोड हो रहा है..."):
            model = pipeline(
                "text-generation",
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                framework="pt",
                device=-1,  # Use CPU
                return_full_text=True
            )
        return model
    except Exception as e:
        st.error(f"❌ मॉडल लोड नहीं हो सका: {str(e)}")
        return None

generator = load_hindi_model()

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def clean_suggestion(text):
    """Clean and validate suggestion text"""
    if not text:
        return ""
    
    # Remove extra whitespace and punctuation from ends
    cleaned = text.strip()
    
    # Remove leading/trailing punctuation except Hindi punctuation
    cleaned = re.sub(r'^[^\u0900-\u097Fa-zA-Z0-9]+|[^\u0900-\u097Fa-zA-Z0-9।]+$', '', cleaned)
    
    # Must contain at least one letter
    if not re.search(r'[\u0900-\u097Fa-zA-Z]', cleaned):
        return ""
    
    # Must be at least 2 characters
    if len(cleaned) < 2:
        return ""
        
    return cleaned

def extract_suggestions(generated_text, original_text):
    """Extract next word suggestions from generated text"""
    if not generated_text or len(generated_text) <= len(original_text):
        return []
    
    # Get the newly generated part
    new_part = generated_text[len(original_text):].strip()
    if not new_part:
        return []
    
    # Split into words and clean them
    words = new_part.split()
    suggestions = []
    
    for word in words[:6]:  # Take first 6 words
        cleaned = clean_suggestion(word)
        if cleaned and cleaned not in suggestions:
            suggestions.append(cleaned)
    
    return suggestions[:N_SUGGESTIONS]

def generate_predictions(text):
    """Generate word predictions using the model"""
    if not generator or not text:
        return []
    
    try:
        # Use last few words for better context
        words = text.strip().split()
        context = " ".join(words[-6:]) if len(words) > 6 else text.strip()
        
        # Generate multiple sequences
        results = generator(
            context,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=N_SUGGESTIONS + 2,
            do_sample=True,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        # Extract suggestions from all results
        all_suggestions = []
        for result in results:
            suggestions = extract_suggestions(result['generated_text'], context)
            all_suggestions.extend(suggestions)
        
        # Remove duplicates and filter
        unique_suggestions = []
        seen = set()
        
        for suggestion in all_suggestions:
            if (suggestion and 
                suggestion.lower() not in seen and 
                suggestion not in text and  # Don't repeat existing words
                len(suggestion) >= 2):
                
                unique_suggestions.append(suggestion)
                seen.add(suggestion.lower())
        
        return unique_suggestions[:N_SUGGESTIONS]
        
    except Exception as e:
        st.error(f"सुझाव बनाने में समस्या: {str(e)}")
        return []

def should_predict(text):
    """Check if we should show predictions"""
    if not text or len(text.strip()) < 1:
        return False
    
    # Don't predict after sentence endings
    if text.strip().endswith(('।', '.', '!', '?')):
        return False
    
    return True

# ------------------------
# SESSION STATE
# ------------------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "suggestion_history" not in st.session_state:
    st.session_state.suggestion_history = []
if "word_count" not in st.session_state:
    st.session_state.word_count = 0

# ------------------------
# MAIN INTERFACE
# ------------------------

# Text input area - Force refresh when session state changes
input_key = f"main_input_{len(st.session_state.user_text)}_{hash(st.session_state.user_text)}"

user_input = st.text_input(
    "यहाँ हिंदी में लिखना शुरू करें...",
    value=st.session_state.user_text,
    placeholder="उदाहरण: आज मौसम बहुत अच्छा है",
    help="हिंदी में कुछ भी टाइप करें और सुझाव देखें",
    key=input_key
)

# Update session state only if user typed (not from button clicks)
if user_input != st.session_state.user_text:
    st.session_state.user_text = user_input
    st.session_state.word_count = len(user_input.split()) if user_input else 0

# Display current text in a visible format if there's accumulated text
if st.session_state.user_text:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
        border-radius: 12px;
        padding: 15px 20px;
        margin: 10px 0;
        border-left: 4px solid #25D366;
        font-size: 18px;
        color: white;
        font-family: 'Noto Sans Devanagari', Arial, sans-serif;
        line-height: 1.5;
    ">
        <strong style="color: #25D366;">आपका टेक्स्ट:</strong><br>
        {st.session_state.user_text}
    </div>
    """, unsafe_allow_html=True)

# Show current stats
if st.session_state.user_text:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("शब्द", len(st.session_state.user_text.split()))
    with col2:
        st.metric("अक्षर", len(st.session_state.user_text))
    with col3:
        st.metric("वाक्य", len([s for s in re.split(r'[।.!?]+', st.session_state.user_text) if s.strip()]))

# Generate and display suggestions
if should_predict(st.session_state.user_text) and generator:
    with st.spinner("💭 सुझाव तैयार हो रहे हैं..."):
        suggestions = generate_predictions(st.session_state.user_text)
    
    if suggestions:
        st.markdown("""
        <div class="suggestion-container">
            <div class="suggestion-title">
                💡 अगले शब्द के सुझाव:
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create suggestion buttons
        cols = st.columns(len(suggestions))
        
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                # Unique key for each suggestion
                button_key = f"suggest_{i}_{hash(suggestion + st.session_state.user_text)}_{len(st.session_state.user_text)}"
                
                if st.button(
                    f"➕ {suggestion}",
                    key=button_key,
                    help=f"'{suggestion}' जोड़ें",
                    use_container_width=True
                ):
                    # Add suggestion to text
                    current_text = st.session_state.user_text.strip()
                    
                    if not current_text:
                        new_text = suggestion
                    elif current_text.endswith(' '):
                        new_text = current_text + suggestion
                    else:
                        new_text = current_text + ' ' + suggestion
                    
                    # Update session state
                    st.session_state.user_text = new_text
                    st.session_state.suggestion_history.append(suggestion)
                    
                    # Show immediate feedback
                    st.success(f"✅ '{suggestion}' जोड़ा गया!")
                    
                    # Force refresh to update input and show new suggestions
                    time.sleep(0.3)  # Brief pause to show success message
                    st.rerun()

# Action buttons
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🗑️ साफ़ करें", use_container_width=True):
        st.session_state.user_text = ""
        st.session_state.suggestion_history = []
        st.rerun()

with col2:
    if st.button("📝 पूर्ण विराम", use_container_width=True):
        if st.session_state.user_text and not st.session_state.user_text.endswith('।'):
            st.session_state.user_text += "। "
            st.rerun()

with col3:
    if st.button("⏪ पिछला शब्द", use_container_width=True):
        words = st.session_state.user_text.strip().split()
        if words:
            words.pop()
            st.session_state.user_text = " ".join(words) + (" " if words else "")
            st.rerun()

with col4:
    if st.button("🔄 नए सुझाव", use_container_width=True):
        st.rerun()

# Show suggestion history
if st.session_state.suggestion_history:
    with st.expander(f"📋 इस्तेमाल किए गए सुझाव ({len(st.session_state.suggestion_history)})"):
        # Show recent suggestions as badges
        recent = st.session_state.suggestion_history[-10:]
        suggestion_badges = " • ".join([f"`{s}`" for s in reversed(recent)])
        st.markdown(f"**हाल के सुझाव:** {suggestion_badges}")

# Text analysis
if st.session_state.user_text:
    with st.expander("🔍 टेक्स्ट विश्लेषण"):
        sentences = [s.strip() for s in re.split(r'[।.!?]+', st.session_state.user_text) if s.strip()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**कुल वाक्य:** {len(sentences)}")
            if sentences:
                avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
                st.write(f"**औसत शब्द प्रति वाक्य:** {avg_words:.1f}")
        
        with col2:
            if sentences:
                st.write("**वाक्यों की सूची:**")
                for i, sentence in enumerate(sentences, 1):
                    st.write(f"{i}. {sentence}")

# Instructions
st.markdown("""
<div class="stats-container">
    <h4>📚 कैसे इस्तेमाल करें:</h4>
    <ul>
        <li><strong>टाइप करें:</strong> हिंदी में कुछ भी लिखना शुरू करें</li>
        <li><strong>सुझाव चुनें:</strong> दिखाए गए शब्दों पर क्लिक करें</li>
        <li><strong>जारी रखें:</strong> नए सुझाव अपने आप आ जाएंगे</li>
        <li><strong>तेज़ टाइपिंग:</strong> WhatsApp की तरह सुझाव इस्तेमाल करें</li>
    </ul>
    <p><strong>💡 टिप:</strong> बेहतर सुझाव के लिए कम से कम 2-3 शब्द लिखें!</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #8b949e;'>🚀 हिंदी AI के साथ तेज़ टाइपिंग का अनुभव करें</p>", 
    unsafe_allow_html=True
)