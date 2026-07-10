# Test Report: AnKu AI

**Project Name**: AnKu AI – AI Fashion Recommendation System  
**Test Cycle**: Beta Release 1.0  
**Testing Scope**: LLM Intent Extraction, FAISS Retrieval, Recommendation Logic, UI Rendering, Fallback Mechanisms.

---

### TC-01: Basic Outfit Recommendation
- **Scenario**: User requests a general outfit without constraints.
- **Input**: "Suggest a nice outfit for me."
- **Expected Output**: Engine returns 3 diverse outfits containing topwear, bottomwear, and footwear.
- **Actual Output**: 3 unique outfits generated successfully.
- **Status**: ✅ Pass

### TC-02: Gender Specificity (Women)
- **Scenario**: Query explicitly specifies female gender.
- **Input**: "I need a party dress for women."
- **Expected Output**: Only female-tagged products are returned.
- **Actual Output**: FAISS pre-filtered by `gender=women`. Female products returned.
- **Status**: ✅ Pass

### TC-03: Gender Specificity (Men)
- **Scenario**: Query explicitly specifies male gender.
- **Input**: "Suggest a men's office outfit."
- **Expected Output**: Only male-tagged products are returned.
- **Actual Output**: FAISS pre-filtered by `gender=men`. Male products returned.
- **Status**: ✅ Pass

### TC-04: Strict Budget Filtering
- **Scenario**: Query includes a strict numeric budget.
- **Input**: "Show me an outfit under 2000 INR."
- **Expected Output**: The *total* price of the assembled outfit is strictly <= 2000.
- **Actual Output**: Outfit assembled. Total price = 1850.
- **Status**: ✅ Pass

### TC-05: Budget Fallback Trigger
- **Scenario**: No exact matches exist within the budget.
- **Input**: "I need a wedding outfit under 500."
- **Expected Output**: System fails to find exact match, triggers budget relaxation, and returns slightly more expensive alternatives in a separate UI section.
- **Actual Output**: UI shows 0 exact matches and displays 3 "Fallback" outfits with relaxed constraints.
- **Status**: ✅ Pass

### TC-06: Product Search Routing
- **Scenario**: User asks for a specific item, not a full outfit.
- **Input**: "I need shoes under 1000 for men."
- **Expected Output**: Gemini sets `query_intent: find_item`. System bypasses outfit assembly and returns individual footwear products.
- **Actual Output**: 4 individual male shoes under 1000 returned.
- **Status**: ✅ Pass

### TC-07: Occasion Normalization
- **Scenario**: User uses a colloquial occasion not strictly in the dataset.
- **Input**: "Suggest something for date night."
- **Expected Output**: Intent extractor normalizes "date night" to the "party" or "casual" tag to prevent FAISS from returning empty results.
- **Actual Output**: Normalized to "party". Outfits returned.
- **Status**: ✅ Pass

### TC-08: Style Keyword Detection
- **Scenario**: User specifies a style sub-genre.
- **Input**: "I want a streetwear look."
- **Expected Output**: Products tagged with `style=streetwear` receive a high compatibility score boost.
- **Actual Output**: Streetwear items dominated the top recommendations.
- **Status**: ✅ Pass

### TC-09: Gemini API Quota Exhaustion
- **Scenario**: Google Gemini API returns HTTP 429.
- **Input**: (Simulated 429 response) "I need a formal outfit."
- **Expected Output**: System catches exception, triggers rule-based regex fallback, extracts 'formal', and proceeds without crashing.
- **Actual Output**: Fallback triggered. UI remained responsive.
- **Status**: ✅ Pass

### TC-10: Gemini API Timeout
- **Scenario**: API takes longer than 15 seconds to respond.
- **Input**: (Simulated network delay)
- **Expected Output**: Request times out, fallback logic engages immediately.
- **Actual Output**: Fallback logic engaged after timeout.
- **Status**: ✅ Pass

### TC-11: Empty Query Handling
- **Scenario**: User submits an empty string.
- **Input**: `""`
- **Expected Output**: Streamlit blocks submission or backend returns a default generalized recommendation.
- **Actual Output**: Streamlit `st.chat_input` naturally blocked empty submission.
- **Status**: ✅ Pass

### TC-12: Gibberish Input
- **Scenario**: User types non-fashion gibberish.
- **Input**: "asdfhjkl"
- **Expected Output**: System searches FAISS, finds low confidence scores, returns safe defaults.
- **Actual Output**: Returned generic casual outfits with low match percentages.
- **Status**: ✅ Pass

### TC-13: Diversity Algorithm Validation
- **Scenario**: Check if multiple generated outfits look identical.
- **Input**: "Suggest 3 office outfits."
- **Expected Output**: The exact same white shirt should not appear in all 3 outfits. Diversity penalty should force alternative topwear.
- **Actual Output**: 3 different shirts recommended across the 3 outfits.
- **Status**: ✅ Pass

### TC-14: Image Loading Failure
- **Scenario**: A product's image file is deleted from the `data/` folder.
- **Input**: "Show me black jeans."
- **Expected Output**: PIL throws `FileNotFoundError`, system catches it, returns `None`, and Streamlit renders the CSS `👕` placeholder instead of crashing.
- **Actual Output**: Placeholder rendered perfectly.
- **Status**: ✅ Pass

### TC-15: Chat History Persistence
- **Scenario**: User asks multiple sequential questions.
- **Input**: Q1: "Need shoes", Q2: "What about a shirt?"
- **Expected Output**: Both queries and both AI responses remain visible in the UI chat log.
- **Actual Output**: Chat history maintained in `st.session_state.messages`.
- **Status**: ✅ Pass

### TC-16: UI Markdown Indentation Bug
- **Scenario**: Check for raw HTML rendering on screen.
- **Input**: Trigger outfit recommendation.
- **Expected Output**: Glassmorphism cards render correctly. Zero visible `<div style="...">` text.
- **Actual Output**: HTML successfully injected via `unsafe_allow_html=True` with no Markdown interference.
- **Status**: ✅ Pass

### TC-17: Sidebar Profile Override
- **Scenario**: User sets profile to "Women" but types "men".
- **Input**: Profile: Women. Query: "Shoes for men."
- **Expected Output**: Query intent ("men") overrides the sidebar profile ("women") based on backend prioritization logic.
- **Actual Output**: Men's shoes returned.
- **Status**: ✅ Pass

### TC-18: Dataset Stats Rendering
- **Scenario**: User navigates to the "Collection Overview" tab.
- **Input**: Click tab.
- **Expected Output**: Data loader reads CSV, calculates metrics, and Streamlit renders 4 metric cards and 4 bar charts.
- **Actual Output**: Charts rendered accurately in < 1 second.
- **Status**: ✅ Pass

### TC-19: Explanations Generation
- **Scenario**: Verify the styling rationale text.
- **Input**: "Outfit for a winter vacation."
- **Expected Output**: The explanation text explicitly mentions the items provided and connects them to "winter vacation".
- **Actual Output**: Explanation generated successfully bridging items to the occasion.
- **Status**: ✅ Pass

### TC-20: Missing API Key Handling
- **Scenario**: `GEMINI_API_KEY` is not set in `.env`.
- **Input**: Application startup.
- **Expected Output**: Application boots normally, sets `gemini_available = False`, and exclusively uses local regex rules. No crash.
- **Actual Output**: Booted successfully. Local engine handled all queries.
- **Status**: ✅ Pass
