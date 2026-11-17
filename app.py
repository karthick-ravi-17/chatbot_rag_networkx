# app.py
import os
import re
import json
import time
import pickle
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
import networkx as nx
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
GRAPH_CACHE_FILE = "tn_agri_schemes_graph.gpickle"
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

st.set_page_config(
    page_title="TN Farmer Schemes Assistant | தமிழ்நாடு விவசாயத் திட்டங்கள்",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for an attractive, agriculture-themed Streamlit UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #388e3c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.9rem;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        font-size: 1rem;
        color: #e8f5e9;
    }
    .scheme-badge {
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #a5d6a7;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. WEB SCRAPER (LANDING + 54 SCHEME PAGES)
# ==========================================
def get_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,ta;q=0.8",
    })
    return session


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def scrape_scheme_detail_page(session: requests.Session, url: str) -> Dict[str, Any]:
    """Scrapes individual scheme detail pages and extracts structured key-value tables & paragraphs."""
    try:
        resp = session.get(url, timeout=20, verify=True)
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove navigation, scripts, styles, and footers to reduce noise
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        details = {}
        raw_sections = []

        # Extract structured table rows (common on tn.gov.in scheme detail pages)
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                cells = [clean_text(td.get_text(" ", strip=True)) for td in tr.find_all(["th", "td"])]
                cells = [c for c in cells if c]
                if len(cells) >= 2:
                    key = cells[0].rstrip(":")
                    val = " | ".join(cells[1:])
                    if len(key) < 120 and val:
                        details[key] = val
                        raw_sections.append(f"{key}: {val}")
                elif len(cells) == 1 and len(cells[0]) > 15:
                    raw_sections.append(cells[0])

        # Fallback / supplement with main content container paragraphs
        main_content = soup.find("div", id=re.compile(r"content|main", re.I)) or soup.body or soup
        for elem in main_content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            txt = clean_text(elem.get_text(" ", strip=True))
            if len(txt) > 15 and txt not in raw_sections:
                raw_sections.append(txt)

        full_text = "\n".join(raw_sections)
        return {
            "structured_fields": details,
            "full_text": full_text[:8000]  # Cap to prevent extreme token bloat
        }
    except Exception as e:
        return {
            "structured_fields": {},
            "full_text": f"Error scraping details: {str(e)}"
        }


def scrape_all_schemes(progress_bar=None, status_text=None) -> Dict[str, Any]:
    """Scrapes the landing page (dep_id=Mg==) and follows all scheme links (expected 54 schemes)."""
    session = get_http_session()
    if status_text:
        status_text.text("Fetching landing page: tn.gov.in/scheme_list.php?dep_id=Mg== ...")

    resp = session.get(BASE_URL, timeout=25)
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    scheme_links = []
    seen_urls = set()

    # Locate scheme detail links on the landing page
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        title = clean_text(a_tag.get_text(" ", strip=True))
        if not title or len(title) < 4:
            continue

        # tn.gov.in scheme detail links typically point to scheme_detail.php or similar scheme pages
        if "scheme" in href.lower() and "scheme_list.php" not in href.lower():
            full_url = urljoin(BASE_URL, href)
            if full_url not in seen_urls:
                seen_urls.add(full_url)
                scheme_links.append({"title": title, "url": full_url})

    total = len(scheme_links)
    schemes_data = []

    for idx, item in enumerate(scheme_links):
        if status_text:
            status_text.text(f"Scraping scheme ({idx + 1}/{total}): {item['title'][:65]}...")
        if progress_bar:
            progress_bar.progress((idx + 1) / max(total, 1))

        page_data = scrape_scheme_detail_page(session, item["url"])
        schemes_data.append({
            "id": f"scheme_{idx + 1}",
            "title": item["title"],
            "url": item["url"],
            "structured_fields": page_data["structured_fields"],
            "full_text": page_data["full_text"]
        })
        time.sleep(0.15)  # Polite crawl delay

    return {
        "department": "Agriculture & Farmers Welfare Department (TN Govt)",
        "landing_url": BASE_URL,
        "total_schemes": len(schemes_data),
        "schemes": schemes_data
    }


# ==========================================
# 3. NETWORKX KNOWLEDGE GRAPH + EMBEDDINGS
# ==========================================
def get_embedding(client: OpenAI, text: str) -> List[float]:
    text = text.replace("\n", " ")[:8000]
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def extract_entities_for_graph(scheme: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extracts lightweight categories/beneficiaries/benefits for graph edges."""
    text_lower = (scheme["title"] + " " + scheme["full_text"]).lower()
    beneficiaries = []
    if any(w in text_lower for w in ["small", "marginal", "சிறு", "குறு"]):
        beneficiaries.append("Small & Marginal Farmers")
    if any(w in text_lower for w in ["women", "female", "மகளிர்"]):
        beneficiaries.append("Women Farmers")
    if any(w in text_lower for w in ["sc/st", "scheduled", "ஆதிதிராவிடர்"]):
        beneficiaries.append("SC/ST Farmers")
    if not beneficiaries:
        beneficiaries.append("All Farmers")

    categories = []
    category_keywords = {
        "Subsidy & Financial Aid": ["subsidy", "assistance", "financial", "grant", "மானியம்"],
        "Irrigation & Water": ["irrigation", "drip", "sprinkler", "water", "பாசனம்", "சொட்டு நீர்"],
        "Seeds & Fertilizers": ["seed", "fertilizer", "nutrient", "organic", "விதை", "உரம்"],
        "Farm Mechanization": ["machinery", "tractor", "equipment", "tools", "இயந்திரம்"],
        "Crop Insurance & Relief": ["insurance", "compensation", "relief", "drought", "காப்பீடு"],
        "Horticulture & Crops": ["horticulture", "paddy", "millet", "sugarcane", "தோட்டக்கலை"]
    }
    for cat, kws in category_keywords.items():
        if any(k in text_lower for k in kws):
            categories.append(cat)
    if not categories:
        categories.append("General Agriculture Welfare")

    return {"beneficiaries": beneficiaries, "categories": categories}


def build_networkx_graph(scraped_data: Dict[str, Any], client: OpenAI, progress_bar=None, status_text=None) -> nx.DiGraph:
    """Builds a NetworkX directed graph storing Department, Schemes, Categories, Beneficiaries, and Embeddings."""
    G = nx.DiGraph()

    dept_node = "DEPT_AGRICULTURE_TN"
    G.add_node(
        dept_node,
        node_type="Department",
        name=scraped_data["department"],
        url=scraped_data["landing_url"],
        total_schemes=scraped_data["total_schemes"]
    )

    schemes = scraped_data["schemes"]
    total = len(schemes)

    for idx, scheme in enumerate(schemes):
        if status_text:
            status_text.text(f"Indexing into NetworkX Graph ({idx + 1}/{total}): {scheme['title'][:55]}...")
        if progress_bar:
            progress_bar.progress((idx + 1) / max(total, 1))

        s_id = scheme["id"]
        rich_document = (
            f"Scheme Title: {scheme['title']}\n"
            f"Source URL: {scheme['url']}\n"
            f"Details:\n{scheme['full_text']}"
        )

        # Compute semantic embedding for hybrid Graph + Vector RAG
        try:
            embedding = get_embedding(client, rich_document)
        except Exception:
            embedding = []

        G.add_node(
            s_id,
            node_type="Scheme",
            title=scheme["title"],
            url=scheme["url"],
            structured_fields=scheme["structured_fields"],
            full_text=scheme["full_text"],
            rich_document=rich_document,
            embedding=embedding
        )

        # Link Department -> Scheme
        G.add_edge(dept_node, s_id, relation="OFFERS_SCHEME")

        # Link Scheme -> Categories & Beneficiaries
        entities = extract_entities_for_graph(scheme)
        for cat in entities["categories"]:
            cat_id = f"CAT_{cat}"
            if not G.has_node(cat_id):
                G.add_node(cat_id, node_type="Category", name=cat)
            G.add_edge(s_id, cat_id, relation="BELONGS_TO_CATEGORY")

        for ben in entities["beneficiaries"]:
            ben_id = f"BEN_{ben}"
            if not G.has_node(ben_id):
                G.add_node(ben_id, node_type="Beneficiary", name=ben)
            G.add_edge(s_id, ben_id, relation="ELIGIBLE_BENEFICIARY")

    with open(GRAPH_CACHE_FILE, "wb") as f:
        pickle.dump(G, f)

    return G


def load_cached_graph() -> nx.DiGraph | None:
    if os.path.exists(GRAPH_CACHE_FILE):
        with open(GRAPH_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return None


# ==========================================
# 4. GRAPH-RAG RETRIEVAL & BILINGUAL LLM
# ==========================================
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def retrieve_from_graph(G: nx.DiGraph, client: OpenAI, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs hybrid retrieval (Vector Cosine Similarity + Keyword + Graph Neighbor Expansion)."""
    query_emb = get_embedding(client, query)
    query_terms = set(re.findall(r'\w+', query.lower()))

    scored_schemes = []
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") != "Scheme":
            continue

        # 1. Semantic similarity score
        sem_score = cosine_similarity(query_emb, data.get("embedding", []))

        # 2. Exact keyword overlap boost (helps with scheme acronyms/Tamil terms)
        doc_lower = data.get("rich_document", "").lower()
        keyword_hits = sum(1 for term in query_terms if len(term) > 2 and term in doc_lower)
        kw_score = min(keyword_hits * 0.05, 0.30)

        total_score = sem_score + kw_score

        # 3. Traverse NetworkX edges to collect connected Categories & Beneficiaries
        neighbors = list(G.successors(node_id))
        categories = [G.nodes[n]["name"] for n in neighbors if G.nodes[n].get("node_type") == "Category"]
        beneficiaries = [G.nodes[n]["name"] for n in neighbors if G.nodes[n].get("node_type") == "Beneficiary"]

        # Find related schemes sharing the same category via graph traversal
        related_titles = set()
        for n in neighbors:
            if G.nodes[n].get("node_type") == "Category":
                for pred in G.predecessors(n):
                    if pred != node_id and G.nodes[pred].get("node_type") == "Scheme":
                        related_titles.add(G.nodes[pred]["title"])

        scored_schemes.append({
            "node_id": node_id,
            "score": total_score,
            "title": data["title"],
            "url": data["url"],
            "full_text": data["full_text"],
            "categories": categories,
            "beneficiaries": beneficiaries,
            "related_schemes": list(related_titles)[:3]
        })

    scored_schemes.sort(key=lambda x: x["score"], reverse=True)
    return scored_schemes[:top_k]


def generate_bilingual_response(
    client: OpenAI,
    user_query: str,
    retrieved_nodes: List[Dict[str, Any]],
    language_mode: str,
    chat_history: List[Dict[str, str]]
) -> str:
    """Generates accurate, farmer-friendly answers in Tamil, English, or Bilingual mode."""
    context_blocks = []
    for idx, item in enumerate(retrieved_nodes, 1):
        block = (
            f"--- SCHEME {idx} ---\n"
            f"Title: {item['title']}\n"
            f"Official Link: {item['url']}\n"
            f"Categories: {', '.join(item['categories'])}\n"
            f"Eligible Beneficiaries: {', '.join(item['beneficiaries'])}\n"
            f"Related Graph Schemes: {', '.join(item['related_schemes'])}\n"
            f"Scraped Details:\n{item['full_text']}\n"
        )
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    lang_instruction = {
        "Bilingual (தமிழ் + English)": (
            "Respond clearly in BOTH Tamil (தமிழ்) and English. "
            "Provide the Tamil explanation first for easy reading by Tamil Nadu farmers, "
            "followed by the English version."
        ),
        "Tamil Only (தமிழ் மட்டும்)": (
            "Respond strictly in clear, respectful, easy-to-understand Tamil (தமிழ்) suitable for farmers."
        ),
        "English Only": (
            "Respond strictly in clear, structured English."
        )
    }.get(language_mode, "Respond in both Tamil and English.")

    system_prompt = f"""You are an official, empathetic, and knowledgeable AI Assistant for Tamil Nadu Farmers, built on data scraped from the Tamil Nadu Government Agriculture Schemes portal (tn.gov.in).

LANGUAGE INSTRUCTION:
{lang_instruction}

STRICT GROUNDING RULES:
1. Answer the farmer's question using ONLY the provided Graph-RAG Scheme Context below.
2. Clearly highlight:
   - Scheme Availability / Status (whether available in the scraped list)
   - Eligibility Criteria (தகுதிகள்)
   - Benefits / Subsidy Details (திட்டத்தின் பயன்கள் / மானியம்)
   - How to Apply / Contact Office (விண்ணப்பிக்கும் முறை)
   - Direct Official Source URL from the context.
3. If the user asks about a scheme or benefit not found in the retrieved context, politely state that it is not listed among the scraped Tamil Nadu Agriculture schemes and suggest the closest related schemes from the context.
4. Keep formatting clean with bullet points and bold labels so farmers can read it easily.

RETRIEVED NETWORKX GRAPH CONTEXT:
{context_str}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content


# ==========================================
# 5. STREAMLIT BILINGUAL UI
# ==========================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌾 தமிழ்நாடு விவசாயிகள் நலத்திட்ட உதவியாளர் | TN Farmer Schemes Chatbot</h1>
        <p>Graph-RAG Powered Bilingual Assistant (NetworkX + OpenAI) | மூலத் தரவு: tn.gov.in</p>
    </div>
    """, unsafe_allow_html=True)

    # Check API Key
    if not OPENAI_API_KEY:
        st.error("⚠️ `OPENAI_API_KEY` not found! Please create a `.env` file with `OPENAI_API_KEY=sk-...` and restart.")
        st.stop()

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Sidebar Controls & Graph Statistics
    with st.sidebar:
        st.header("⚙️ Settings / அமைப்புகள்")
        language_mode = st.selectbox(
            "Select Response Language / மொழியைத் தேர்ந்தெடுக்கவும்:",
            [
                "Bilingual (தமிழ் + English)",
                "Tamil Only (தமிழ் மட்டும்)",
                "English Only"
            ],
            index=0
        )

        st.divider()
        st.subheader("🕸️ NetworkX Knowledge Graph")

        G = load_cached_graph()

        if st.button("🔄 Scrape & Rebuild Graph (55 Pages)", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            scraped_data = scrape_all_schemes(progress_bar, status_text)
            G = build_networkx_graph(scraped_data, client, progress_bar, status_text)
            status_text.success(f"✅ Indexed {scraped_data['total_schemes']} schemes into NetworkX!")
            st.rerun()

        if G is None:
            st.info("First-time setup: Scraping tn.gov.in landing page + 54 scheme pages...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            scraped_data = scrape_all_schemes(progress_bar, status_text)
            G = build_networkx_graph(scraped_data, client, progress_bar, status_text)
            status_text.empty()
            progress_bar.empty()
            st.rerun()

        # Display NetworkX Graph Metrics
        scheme_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Scheme"]
        cat_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Category"]

        col1, col2 = st.columns(2)
        col1.metric("Schemes (Nodes)", len(scheme_nodes))
        col2.metric("Graph Edges", G.number_of_edges())
        st.caption(f"Categories Mapped: {len(cat_nodes)} | Source: [tn.gov.in]({BASE_URL})")

        with st.expander("📋 Browse All Scraped Schemes"):
            for idx, n_id in enumerate(scheme_nodes, 1):
                st.markdown(f"**{idx}.** [{G.nodes[n_id]['title']}]({G.nodes[n_id]['url']})")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Quick-Ask Sample Questions for Farmers
    st.markdown("**💡 விரைவு கேள்விகள் / Quick Sample Questions:**")
    q_cols = st.columns(3)
    sample_questions = [
        "சொட்டு நீர் பாசனத்திற்கு என்ன மானியம் உள்ளது? (Drip irrigation subsidy details)",
        "சிறு மற்றும் குறு விவசாயிகளுக்கான திட்டங்கள் என்ன? (Schemes for small & marginal farmers)",
        "விவசாய இயந்திரங்கள் வாங்க மானியம் கிடைக்குமா? (Farm machinery subsidy availability)"
    ]
    selected_sample = None
    for i, q in enumerate(sample_questions):
        if q_cols[i].button(q, use_container_width=True):
            selected_sample = q

    st.divider()

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "வணக்கம்! 🙏 தமிழ்நாடு அரசின் வேளாண்மைத் துறை திட்டங்கள் குறித்து நீங்கள் எதை வேண்டுமானாலும் கேட்கலாம்.\n\n"
                    "Welcome! Ask me about any Tamil Nadu Government Agriculture scheme availability, subsidies, eligibility, or application details."
                )
            }
        ]

    # Render Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle Input (either from quick sample button or chat input box)
    user_prompt = st.chat_input("உங்கள் கேள்வியை தட்டச்சு செய்க... / Type your question about TN schemes...")
    if selected_sample:
        user_prompt = selected_sample

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching NetworkX Knowledge Graph & generating response..."):
                retrieved = retrieve_from_graph(G, client, user_prompt, top_k=5)
                answer = generate_bilingual_response(
                    client=client,
                    user_query=user_prompt,
                    retrieved_nodes=retrieved,
                    language_mode=language_mode,
                    chat_history=st.session_state.messages[:-1]
                )
                st.markdown(answer)

                # Show Graph Provenance / Retrieved Nodes
                with st.expander("🔍 Graph-RAG Sources & Related Nodes Used"):
                    for item in retrieved:
                        st.markdown(
                            f"- **{item['title']}** (Score: `{item['score']:.2f}`) | "
                            f"Categories: `{', '.join(item['categories'])}` | "
                            f"[Official Page]({item['url']})"
                        )

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()