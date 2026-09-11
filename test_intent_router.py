import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the 6 Canonical Intents & Actions
INTENT_GENERAL_CHAT = "GENERAL_CHAT"
INTENT_PROMPT_WRITING = "PROMPT_WRITING"
INTENT_IMAGE_GENERATION = "IMAGE_GENERATION"
INTENT_IMAGE_ANALYSIS = "IMAGE_ANALYSIS"
INTENT_FILE_ANALYSIS = "FILE_ANALYSIS"
INTENT_CODE_GENERATION = "CODE_GENERATION"

ACTION_CHAT = "CHAT"
ACTION_WRITE_PROMPT = "WRITE_PROMPT"
ACTION_GENERATE_IMAGE = "GENERATE_IMAGE"
ACTION_ANALYZE_IMAGE = "ANALYZE_IMAGE"
ACTION_ANALYZE_FILE = "ANALYZE_FILE"
ACTION_GENERATE_CODE = "GENERATE_CODE"
ACTION_BLOCK = "BLOCK_REQUEST"

# Curated Intent Corpus for Supporting / Fallback Vector Space
INTENT_CORPUS = [
    # GENERAL_CHAT
    ("explain artificial intelligence", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("what is machine learning and deep learning", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("how does photosynthesis work in plants", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("who was alan turing and what did he invent", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("tell me about the history of the internet", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("what is the capital of france", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("how can i generate a car image", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("what tools can be used to create ai images", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("explain prompt injection attacks and defense strategies", INTENT_GENERAL_CHAT, ACTION_CHAT),
    ("what is malware and how do firewalls work", INTENT_GENERAL_CHAT, ACTION_CHAT),

    # PROMPT_WRITING
    ("give me a prompt to generate a realistic red sports car", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("i need a prompt for generating a car image in chatgpt", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("give me a prompt for a car image", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("write a midjourney prompt for a futuristic cyberpunk city", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("suggest a dall-e 3 prompt for a cozy coffee shop", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("provide a detailed prompt for generating an image of a blue dragon", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("craft an image prompt for a fantasy landscape", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("give me prompt ideas for stable diffusion", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("make it cinematic", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("refine the prompt to be more dramatic with neon lighting", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),
    ("add golden hour lighting to the prompt", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT),

    # IMAGE_GENERATION
    ("generate a realistic red sports car on a mountain road", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("create an image of a blue bus", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("generate an image of a snake in a forest", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("draw a picture of an astronaut riding a horse on mars", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("render a 3d isometric cyberpunk bedroom", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("produce a realistic photo of a golden retriever puppy", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("paint a watercolor landscape of snowy mountains at dawn", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("now generate the image", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("now generate it", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),
    ("generate the car image now", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE),

    # IMAGE_ANALYSIS
    ("what is in this image", INTENT_IMAGE_ANALYSIS, ACTION_ANALYZE_IMAGE),
    ("describe what you see in this picture", INTENT_IMAGE_ANALYSIS, ACTION_ANALYZE_IMAGE),
    ("analyze this screenshot and tell me if there are errors", INTENT_IMAGE_ANALYSIS, ACTION_ANALYZE_IMAGE),
    ("explain the objects in this photo", INTENT_IMAGE_ANALYSIS, ACTION_ANALYZE_IMAGE),
    ("extract the text from this image", INTENT_IMAGE_ANALYSIS, ACTION_ANALYZE_IMAGE),

    # FILE_ANALYSIS
    ("summarize this pdf", INTENT_FILE_ANALYSIS, ACTION_ANALYZE_FILE),
    ("analyze this document and highlight key findings", INTENT_FILE_ANALYSIS, ACTION_ANALYZE_FILE),
    ("what does this uploaded csv file contain", INTENT_FILE_ANALYSIS, ACTION_ANALYZE_FILE),
    ("summarize the attached report", INTENT_FILE_ANALYSIS, ACTION_ANALYZE_FILE),
    ("explain this spreadsheet data", INTENT_FILE_ANALYSIS, ACTION_ANALYZE_FILE),

    # CODE_GENERATION
    ("write python code for a calculator", INTENT_CODE_GENERATION, ACTION_GENERATE_CODE),
    ("can you give me python code to check if a number is prime or not", INTENT_CODE_GENERATION, ACTION_GENERATE_CODE),
    ("create a javascript function to validate email addresses", INTENT_CODE_GENERATION, ACTION_GENERATE_CODE),
    ("implement a binary search algorithm in java", INTENT_CODE_GENERATION, ACTION_GENERATE_CODE),
    ("write a sql query to find top paying customers", INTENT_CODE_GENERATION, ACTION_GENERATE_CODE),
    ("give me a bash script to backup a postgresql database", INTENT_CODE_GENERATION, ACTION_GENERATE_CODE),
]

_CORPUS_TEXTS = [item[0] for item in INTENT_CORPUS]
_INTENT_VECTORIZER = TfidfVectorizer(ngram_range=(1, 3), analyzer="word").fit(_CORPUS_TEXTS)
_INTENT_MATRIX = _INTENT_VECTORIZER.transform(_CORPUS_TEXTS)


def classify_intent_fallback(user_prompt: str, history_messages=None, has_file: bool = False, has_image: bool = False) -> dict:
    """
    Syntactic & Semantic Fallback Router used when primary LLM classifier is unavailable or times out.
    Evaluates complete sentence grammar, objects, context, and TF-IDF supporting vector distance.
    """
    p_clean = user_prompt.strip()
    p_lower = p_clean.lower()
    
    # 1. Check Multi-Action ("Give me a prompt for X, then generate it")
    has_prompt_req = bool(re.search(r"\b(give|write|need|suggest|create|draft|provide)\b.*\b(prompt)\b", p_lower))
    has_gen_followup = bool(re.search(r"\b(then|afterwards|now|and)\s+(generate|create|render|draw|make)\s+(it|the\s+image|image|photo)\b", p_lower))
    if has_prompt_req and has_gen_followup:
        return {
            "intent": INTENT_IMAGE_GENERATION,
            "confidence": 0.96,
            "action": ACTION_GENERATE_IMAGE,
            "secondary_action": ACTION_WRITE_PROMPT,
            "is_multi_action": True,
            "reasoning": "User requested prompt writing followed sequentially by image generation."
        }

    # 2. File / Image Analysis based on uploads or explicit directives
    if has_image or ("image" in p_lower and any(w in p_lower for w in ["what is in", "describe this", "analyze this", "inspect this", "what does this image show", "read this image"])):
        return {
            "intent": INTENT_IMAGE_ANALYSIS,
            "confidence": 0.95,
            "action": ACTION_ANALYZE_IMAGE,
            "secondary_action": None,
            "is_multi_action": False,
            "reasoning": "User requested analysis of an image or uploaded visual media."
        }
    
    if has_file or any(w in p_lower for w in ["summarize this pdf", "summarize the pdf", "analyze this file", "analyze the document", "summarize this document", "read this file", "what is in this csv", "summarize this report"]):
        return {
            "intent": INTENT_FILE_ANALYSIS,
            "confidence": 0.95,
            "action": ACTION_ANALYZE_FILE,
            "secondary_action": None,
            "is_multi_action": False,
            "reasoning": "User requested document or file analysis/summarization."
        }

    # 3. Contextual Multi-Turn Continuation
    if history_messages and len(history_messages) > 0:
        last_intent = None
        for msg in reversed(history_messages):
            if msg.get("role") == "assistant" and "res" in msg:
                last_intent = msg["res"].get("detected_intent") or msg["res"].get("intent")
                break
        
        # If user says "Now generate the image", "now generate it", "create the image now"
        if re.search(r"^\b(now|please)?\s*(generate|render|draw)\s+(the\s+image|it|the\s+picture)?\b", p_lower) or re.search(r"\b(generate|create|render|draw)\s+(the\s+image|the\s+picture)\b", p_lower) or re.search(r"\bnow\s+(generate|create|render|draw)\b", p_lower):
            if not any(w in p_lower for w in ["prompt", "ideas", "how"]):
                return {
                    "intent": INTENT_IMAGE_GENERATION,
                    "confidence": 0.97,
                    "action": ACTION_GENERATE_IMAGE,
                    "secondary_action": None,
                    "is_multi_action": False,
                    "reasoning": "User requested to transition from previous prompt writing to generating the actual image."
                }

        # If user says "Make it cinematic", "add neon lights", "more realistic", and previous intent was PROMPT_WRITING
        if last_intent in [INTENT_PROMPT_WRITING, "PROMPT_WRITING"] or any("prompt" in str(msg.get("content", "")).lower() for msg in history_messages[-2:]):
            if any(p_lower.startswith(w) for w in ["make it", "add ", "change it", "make the", "in ", "with "]) or "cinematic" in p_lower or "realistic" in p_lower or "lighting" in p_lower:
                if not any(w in p_lower for w in ["now generate", "actually generate", "create the image"]):
                    return {
                        "intent": INTENT_PROMPT_WRITING,
                        "confidence": 0.95,
                        "action": ACTION_WRITE_PROMPT,
                        "secondary_action": None,
                        "is_multi_action": False,
                        "reasoning": "User is continuing multi-turn prompt refinement to adjust prompt details."
                    }

    # 4. Explicit PROMPT_WRITING Check
    prompt_asking_patterns = [
        r"\b(give|provide|show|write|craft|suggest|need|want|draft|generate|create)\b.*\b(prompt|prompts)\b",
        r"\b(prompt|prompts)\s+(to|for|about)\s+(generate|creating|generating|drawing|make|making)",
        r"\b(image\s+prompt|midjourney\s+prompt|dall-?e\s+prompt|diffusion\s+prompt)\b",
        r"\bprompt\s+ideas?\b",
    ]
    is_asking_for_prompt = any(re.search(pat, p_lower) for pat in prompt_asking_patterns)
    if is_asking_for_prompt:
        return {
            "intent": INTENT_PROMPT_WRITING,
            "confidence": 0.98,
            "action": ACTION_WRITE_PROMPT,
            "secondary_action": None,
            "is_multi_action": False,
            "reasoning": "User explicitly asked to craft or write a prompt, not to generate an image."
        }

    # 5. CODE_GENERATION Check
    code_patterns = [
        r"\b(python|javascript|typescript|c\+\+|java|rust|go|sql|bash|powershell|html|css)\s+(code|script|program|function|algorithm|class)\b",
        r"\b(write|give\s+me|create|implement|draft)\b.*\b(code|script|function|algorithm|program)\b",
        r"\bcode\s+(for|to|that)\b",
        r"\b(check\s+if\s+a\s+number\s+is\s+prime|calculator\s+in\s+python|binary\s+search)\b",
    ]
    if any(re.search(pat, p_lower) for pat in code_patterns):
        return {
            "intent": INTENT_CODE_GENERATION,
            "confidence": 0.96,
            "action": ACTION_GENERATE_CODE,
            "secondary_action": None,
            "is_multi_action": False,
            "reasoning": "User requested programming code generation."
        }

    # 6. Direct IMAGE_GENERATION Check (imperative to generate/create an image)
    is_educational_how = bool(re.search(r"^\b(how\s+(can|do|to)\s+(i|we)?\s*(generate|create|make))\b", p_lower))
    if not is_educational_how:
        image_gen_patterns = [
            r"^\b(generate|create|render|draw|produce|paint)\s+(an?|some)?\s*(realistic|photorealistic|cinematic|detailed|3d)?\s*(image|photo|picture|wallpaper|render|illustration|portrait)\b",
            r"^\b(generate|create|render|draw)\s+(an?|some)?\s*(\w+\s+)*(car|bus|snake|dog|cat|bird|mountains?|city|forest|landscape|dragon|robot|astronaut)\b",
            r"\b(generate|create|draw|render)\s+a\s+realistic\s+[a-z\s]+(on|in|at|with)\b"
        ]
        if any(re.search(pat, p_lower) for pat in image_gen_patterns):
            return {
                "intent": INTENT_IMAGE_GENERATION,
                "confidence": 0.96,
                "action": ACTION_GENERATE_IMAGE,
                "secondary_action": None,
                "is_multi_action": False,
                "reasoning": "User directly commanded the creation/rendering of an image."
            }

    # 7. TF-IDF Supporting Cosine Similarity
    vec = _INTENT_VECTORIZER.transform([user_prompt])
    sims = cosine_similarity(vec, _INTENT_MATRIX)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    matched_text, corpus_intent, corpus_action = INTENT_CORPUS[best_idx]

    if best_score >= 0.40:
        return {
            "intent": corpus_intent,
            "confidence": round(min(best_score + 0.35, 0.98), 2),
            "action": corpus_action,
            "secondary_action": None,
            "is_multi_action": False,
            "reasoning": f"TF-IDF supporting match ({best_score:.2f}) with benchmark '{matched_text}'."
        }

    # Default to GENERAL_CHAT
    return {
        "intent": INTENT_GENERAL_CHAT,
        "confidence": 0.90,
        "action": ACTION_CHAT,
        "secondary_action": None,
        "is_multi_action": False,
        "reasoning": "Standard informational or general conversational inquiry."
    }


def classify_intent_with_llm(user_prompt: str, history_messages=None, has_file: bool = False, has_image: bool = False, engine_choice: str = "", api_key: str = "") -> dict:
    """
    Primary LLM-based Semantic Intent Classifier with structured JSON output.
    Falls back to `classify_intent_fallback` only if LLM is unreachable or disabled.
    """
    system_prompt = (
        "You are an expert Semantic Intent Classifier for an AI Security Gateway.\n"
        "Analyze the user's complete prompt, multi-turn conversation context, and uploaded file status.\n"
        "Classify into EXACTLY ONE of these 6 intents:\n"
        "1. GENERAL_CHAT (action: 'CHAT'): Explanations, Q&A, general inquiries (e.g., 'Explain artificial intelligence', 'How can I generate a car image?').\n"
        "2. PROMPT_WRITING (action: 'WRITE_PROMPT'): User wants a prompt to be written/crafted for image/video/text diffusion models (e.g., 'Give me a prompt to generate a realistic red sports car', 'I need a prompt for generating a car image in ChatGPT', 'Make it cinematic'). CRITICAL: Do NOT select IMAGE_GENERATION when user asks FOR a prompt!\n"
        "3. IMAGE_GENERATION (action: 'GENERATE_IMAGE'): User commands AI to generate/render/draw an image itself right now (e.g., 'Generate a realistic red sports car on a mountain road', 'Create an image of a blue bus', 'Now generate the image').\n"
        "4. IMAGE_ANALYSIS (action: 'ANALYZE_IMAGE'): User wants an image inspected/described/analyzed (e.g., 'What is in this image?').\n"
        "5. FILE_ANALYSIS (action: 'ANALYZE_FILE'): User wants a document/PDF/CSV summarized/analyzed (e.g., 'Summarize this PDF').\n"
        "6. CODE_GENERATION (action: 'GENERATE_CODE'): User asks for programming code/functions/scripts (e.g., 'Write Python code for a calculator').\n\n"
        "If the user asks for BOTH (e.g. 'Give me a prompt for a car image, then generate it'), return:\n"
        "intent: 'IMAGE_GENERATION', action: 'GENERATE_IMAGE', secondary_action: 'WRITE_PROMPT', is_multi_action: true.\n\n"
        "Respond ONLY with valid JSON in this exact structure:\n"
        "{\n"
        '  "intent": "PROMPT_WRITING",\n'
        '  "confidence": 0.96,\n'
        '  "action": "WRITE_PROMPT",\n'
        '  "secondary_action": null,\n'
        '  "is_multi_action": false,\n'
        '  "reasoning": "..."\n'
        "}"
    )

    if "OpenAI" in engine_choice and api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            llm_msgs = [{"role": "system", "content": system_prompt}]
            if history_messages:
                for m in history_messages[-4:]:
                    if m.get("role") in ["user", "assistant"]:
                        c = m.get("content") or (m.get("answer", {}).get("content") if isinstance(m.get("answer"), dict) else "")
                        if c:
                            llm_msgs.append({"role": m["role"], "content": str(c)[:300]})
            llm_msgs.append({"role": "user", "content": f"User Prompt: {user_prompt}\nAttached File: {has_file}\nAttached Image: {has_image}"})
            
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=llm_msgs,
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=5.0
            )
            data = json.loads(resp.choices[0].message.content.strip())
            if "intent" in data and "action" in data:
                return data
        except Exception:
            pass

    if "Ollama" in engine_choice:
        try:
            import ollama
            llm_msgs = [{'role': 'system', 'content': system_prompt}]
            if history_messages:
                for m in history_messages[-4:]:
                    if m.get("role") in ["user", "assistant"]:
                        c = m.get("content") or (m.get("answer", {}).get("content") if isinstance(m.get("answer"), dict) else "")
                        if c:
                            llm_msgs.append({'role': m['role'], 'content': str(c)[:300]})
            llm_msgs.append({'role': 'user', 'content': f"User Prompt: {user_prompt}\nAttached File: {has_file}\nAttached Image: {has_image}"})

            resp = ollama.chat(
                model='llama3.2',
                messages=llm_msgs,
                options={'temperature': 0.0}
            )
            raw = resp['message']['content'].strip()
            json_m = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_m:
                data = json.loads(json_m.group(0))
                if "intent" in data and "action" in data:
                    return data
        except Exception:
            pass

    return classify_intent_fallback(user_prompt, history_messages, has_file, has_image)


def resolve_semantic_routing(
    user_prompt: str,
    history_messages=None,
    has_file: bool = False,
    has_image: bool = False,
    security_res: dict = None,
    engine_choice: str = "",
    api_key: str = ""
) -> dict:
    if security_res and security_res.get("action") == "BLOCK":
        return {
            "intent": "SECURITY_BLOCK",
            "confidence": 1.0,
            "action": ACTION_BLOCK,
            "secondary_action": None,
            "is_multi_action": False,
            "reasoning": f"Halted by Security Gateway: {security_res.get('reason')}"
        }

    classification = classify_intent_with_llm(
        user_prompt=user_prompt,
        history_messages=history_messages,
        has_file=has_file,
        has_image=has_image,
        engine_choice=engine_choice,
        api_key=api_key
    )

    if classification["intent"] != INTENT_IMAGE_GENERATION:
        assert classification["action"] != ACTION_GENERATE_IMAGE, (
            f"Invariant violation: Action '{classification['action']}' cannot be GENERATE_IMAGE for intent '{classification['intent']}'"
        )

    return classification


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING INTENT ROUTER TESTS")
    print("=" * 80)

    test_cases = [
        ("Explain artificial intelligence", INTENT_GENERAL_CHAT, ACTION_CHAT, False, False),
        ("Give me a prompt to generate a realistic red sports car", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT, False, False),
        ("I need a prompt for generating a car image in ChatGPT", INTENT_PROMPT_WRITING, ACTION_WRITE_PROMPT, False, False),
        ("Generate a realistic red sports car on a mountain road", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE, False, False),
        ("Create an image of a blue bus", INTENT_IMAGE_GENERATION, ACTION_GENERATE_IMAGE, False, False),
        ("What is in this image?", INTENT_IMAGE_ANALYSIS, ACTION_ANALYZE_IMAGE, False, True),
        ("Summarize this PDF", INTENT_FILE_ANALYSIS, ACTION_ANALYZE_FILE, True, False),
        ("Write Python code for a calculator", INTENT_CODE_GENERATION, ACTION_GENERATE_CODE, False, False),
        ("How can I generate a car image?", INTENT_GENERAL_CHAT, ACTION_CHAT, False, False),
    ]

    all_passed = True
    for prompt, expected_intent, expected_action, h_file, h_img in test_cases:
        res = resolve_semantic_routing(prompt, history_messages=None, has_file=h_file, has_image=h_img)
        passed = (res["intent"] == expected_intent and res["action"] == expected_action)
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"[{status}] \"{prompt}\" -> Intent: {res['intent']} | Action: {res['action']}")

    # Multi-action test
    multi_prompt = "Give me a prompt for a car image, then generate it"
    multi_res = resolve_semantic_routing(multi_prompt)
    multi_passed = multi_res["is_multi_action"] and multi_res["secondary_action"] == ACTION_WRITE_PROMPT
    print(f"[{'PASSED' if multi_passed else 'FAILED'}] \"{multi_prompt}\" -> Multi-Action: {multi_res['is_multi_action']}")
    if not multi_passed:
        all_passed = False

    # Multi-turn Context Test
    print("\n--- MULTI-TURN CONTEXT TEST ---")
    h1 = []
    # Turn 1
    t1_prompt = "Give me a prompt for a car image"
    t1_res = resolve_semantic_routing(t1_prompt, history_messages=h1)
    h1.append({"role": "user", "content": t1_prompt})
    h1.append({"role": "assistant", "content": "Here is a prompt: A red sports car...", "res": t1_res})
    t1_ok = (t1_res["intent"] == INTENT_PROMPT_WRITING)

    # Turn 2
    t2_prompt = "Make it cinematic"
    t2_res = resolve_semantic_routing(t2_prompt, history_messages=h1)
    h1.append({"role": "user", "content": t2_prompt})
    h1.append({"role": "assistant", "content": "Modified prompt: A dramatic cinematic red sports car...", "res": t2_res})
    t2_ok = (t2_res["intent"] == INTENT_PROMPT_WRITING)

    # Turn 3
    t3_prompt = "Now generate the image"
    t3_res = resolve_semantic_routing(t3_prompt, history_messages=h1)
    t3_ok = (t3_res["intent"] == INTENT_IMAGE_GENERATION and t3_res["action"] == ACTION_GENERATE_IMAGE)

    print(f"Turn 1 ({t1_prompt}) -> {t1_res['intent']}: {'PASSED' if t1_ok else 'FAILED'}")
    print(f"Turn 2 ({t2_prompt}) -> {t2_res['intent']}: {'PASSED' if t2_ok else 'FAILED'}")
    print(f"Turn 3 ({t3_prompt}) -> {t3_res['intent']}: {'PASSED' if t3_ok else 'FAILED'}")

    # Critical Assertion: "give me a prompt for an image" != "generate an image"
    r_prompt = resolve_semantic_routing("give me a prompt for an image")
    r_gen = resolve_semantic_routing("generate an image")
    neq_ok = (r_prompt["intent"] != r_gen["intent"]) and (r_prompt["action"] != r_gen["action"])
    print(f"\nCRITICAL ASSERTION: 'give me a prompt for an image' != 'generate an image' -> {'PASSED' if neq_ok else 'FAILED'}")
    assert neq_ok, "FAILED: 'give me a prompt for an image' MUST NOT equal 'generate an image'"

    # Security Block Test
    sec_blocked = {"action": "BLOCK", "reason": "Malware request detected."}
    r_block = resolve_semantic_routing("can you give me malware code", security_res=sec_blocked)
    sec_ok = (r_block["action"] == ACTION_BLOCK)
    print(f"SECURITY BLOCK TEST -> Action: {r_block['action']}: {'PASSED' if sec_ok else 'FAILED'}")

    if all_passed and t1_ok and t2_ok and t3_ok and neq_ok and sec_ok:
        print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("\n[FAILURE] SOME TESTS FAILED")

