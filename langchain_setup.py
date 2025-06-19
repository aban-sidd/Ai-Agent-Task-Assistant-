from langchain.tools import tool
from tavily import TavilyClient
from transformers import pipeline

from pydantic import BaseModel
from langchain.agents import initialize_agent , AgentType , Tool , create_openai_functions_agent , AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import streamlit as st


try:
    # ✅ Pull from Streamlit Cloud Secrets
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

    if not TAVILY_API_KEY or not isinstance(TAVILY_API_KEY, str):
        raise ValueError("❌ TAVILY_API_KEY is missing or invalid.")

    if not OPENAI_API_KEY or not isinstance(OPENAI_API_KEY, str):
        raise ValueError("❌ OPENAI_API_KEY is missing or invalid.")

except Exception as e:
    st.error(f"🔐 Failed to load secrets: {e}")
    st.stop()

## Web-search Tool

print("✅ Loaded secrets")
print("✅ Creating Tavily client...")

client = TavilyClient(api_key=TAVILY_API_KEY)

print("✅ Tavily client ready")
print("✅ Initializing tools...")

@tool
def web_search(query: str) -> str:
  """Searches the web for the given query using Tavily API."""
  results = client.search(query)
  return results['results'][0]['content']

## Summarize tool

@tool
def summarize(text: str) -> str:
  """Summarizes the given text."""
  return 'summarize this: ' +  text

## Translation tool

LANG_CODES = {
    "acehnese arabic": "ace_Arab", "acehnese latin": "ace_Latn",
    "mesopotamian arabic": "acm_Arab", "taizzi‑adeni arabic": "acq_Arab",
    "tunisian arabic": "aeb_Arab", "afrikaans": "afr_Latn",
    "south levantine arabic": "ajp_Arab", "akan": "aka_Latn",
    "amharic": "amh_Ethi", "north levantine arabic": "apc_Arab",
    "modern standard arabic": "arb_Arab", "arabic latin": "arb_Latn",
    "najdi arabic": "ars_Arab", "moroccan arabic": "ary_Arab",
    "egyptian arabic": "arz_Arab", "assamese": "asm_Beng",
    "asturian": "ast_Latn", "awadhi": "awa_Deva",
    "aymara": "ayr_Latn", "south azerbaijani": "azb_Arab",
    "north azerbaijani": "azj_Latn", "bashkir": "bak_Cyrl",
    "bambara": "bam_Latn", "balinese": "ban_Latn",
    "belarusian": "bel_Cyrl", "bemba": "bem_Latn",
    "bengali": "ben_Beng", "bhojpuri": "bho_Deva",
    "banjar arabic": "bjn_Arab", "banjar latin": "bjn_Latn",
    "tibetan": "bod_Tibt", "bosnian": "bos_Latn",
    "buginese": "bug_Latn", "bulgarian": "bul_Cyrl",
    "catalan": "cat_Latn", "cebuano": "ceb_Latn",
    "czech": "ces_Latn", "chokwe": "cjk_Latn",
    "central kurdish": "ckb_Arab", "crimean tatar": "crh_Latn",
    "welsh": "cym_Latn", "danish": "dan_Latn",
    "german": "deu_Latn", "southwestern dinka": "dik_Latn",
    "dyula": "dyu_Latn", "dzongkha": "dzo_Tibt",
    "greek": "ell_Grek", "english": "eng_Latn",
    "esperanto": "epo_Latn", "estonian": "est_Latn",
    "basque": "eus_Latn", "ewe": "ewe_Latn",
    "faroese": "fao_Latn", "fijian": "fij_Latn",
    "finnish": "fin_Latn", "fon": "fon_Latn",
    "french": "fra_Latn", "friulian": "fur_Latn",
    "fulfulde": "fuv_Latn", "scottish gaelic": "gla_Latn",
    "irish": "gle_Latn", "galician": "glg_Latn",
    "guarani": "grn_Latn", "gujarati": "guj_Gujr",
    "haitian creole": "hat_Latn", "hausa": "hau_Latn",
    "hebrew": "heb_Hebr", "hindi": "hin_Deva",
    "chhattisgarhi": "hne_Deva", "croatian": "hrv_Latn",
    "hungarian": "hun_Latn", "armenian": "hye_Armn",
    "igbo": "ibo_Latn", "ilocano": "ilo_Latn",
    "indonesian": "ind_Latn", "icelandic": "isl_Latn",
    "italian": "ita_Latn", "javanese": "jav_Latn",
    "japanese": "jpn_Jpan", "kabyle": "kab_Latn",
    "jingpho": "kac_Latn", "kamba": "kam_Latn",
    "kannada": "kan_Knda", "kashmiri arabic": "kas_Arab",
    "kashmiri devanagari": "kas_Deva", "georgian": "kat_Geor",
    "kanuri arabic": "knc_Arab", "kanuri latin": "knc_Latn",
    "kazakh": "kaz_Cyrl", "kabiyè": "kbp_Latn",
    "kabuverdianu": "kea_Latn", "khmer": "khm_Khmr",
    "kikuyu": "kik_Latn", "kinyarwanda": "kin_Latn",
    "kyrgyz": "kir_Cyrl", "kimbundu": "kmb_Latn",
    "northern kurdish": "kmr_Latn", "kikongo": "kon_Latn",
    "korean": "kor_Hang", "lao": "lao_Laoo",
    "ligurian": "lij_Latn", "limburgish": "lim_Latn",
    "lingala": "lin_Latn", "lithuanian": "lit_Latn",
    "lombard": "lmo_Latn", "latgalian": "ltg_Latn",
    "luxembourgish": "ltz_Latn", "luba‑kasai": "lua_Latn",
    "ganda": "lug_Latn", "luo": "luo_Latn",
    "mizo": "lus_Latn", "latvian": "lvs_Latn",
    "magahi": "mag_Deva", "maithili": "mai_Deva",
    "malayalam": "mal_Mlym", "marathi": "mar_Deva",
    "minangkabau arabic": "min_Arab", "minangkabau latin": "min_Latn",
    "macedonian": "mkd_Cyrl", "plateau malagasy": "plt_Latn",
    "maltese": "mlt_Latn", "meitei bengali": "mni_Beng",
    "mongolian": "khk_Cyrl", "mossi": "mos_Latn",
    "maori": "mri_Latn", "burmese": "mya_Mymr",
    "dutch": "nld_Latn", "norwegian nynorsk": "nno_Latn",
    "norwegian bokmål": "nob_Latn", "nepali": "npi_Deva",
    "northern sotho": "nso_Latn", "nuer": "nus_Latn",
    "nyanja": "nya_Latn", "occitan": "oci_Latn",
    "oromo": "gaz_Latn", "odia": "ory_Orya",
    "panjabi": "pan_Guru", "papiamento": "pap_Latn",
    "persian": "pes_Arab", "polish": "pol_Latn",
    "portuguese": "por_Latn", "dari": "prs_Arab",
    "pashto": "pbt_Arab", "quechua": "quy_Latn",
    "romanian": "ron_Latn", "rundi": "run_Latn",
    "russian": "rus_Cyrl", "sango": "sag_Latn",
    "sanskrit": "san_Deva", "santali": "sat_Olck",
    "sicilian": "scn_Latn", "shan": "shn_Mymr",
    "sinhala": "sin_Sinh", "slovak": "slk_Latn",
    "slovenian": "slv_Latn", "samoan": "smo_Latn",
    "shona": "sna_Latn", "sindhi arabic": "snd_Arab",
    "somali": "som_Latn", "southern sotho": "sot_Latn",
    "spanish": "spa_Latn", "albanian tosk": "als_Latn",
    "sardinian": "srd_Latn", "serbian cyrillic": "srp_Cyrl",
    "swati": "ssw_Latn", "sundanese": "sun_Latn",
    "swedish": "swe_Latn", "swahili": "swh_Latn",
    "silesian": "szl_Latn", "tamil": "tam_Taml",
    "tatar": "tat_Cyrl", "telugu": "tel_Telu",
    "tajik": "tgk_Cyrl", "tagalog": "tgl_Latn",
    "thai": "tha_Thai", "tigrinya": "tir_Ethi",
    "tamasheq latin": "taq_Latn", "tamasheq tifinagh": "taq_Tfng",
    "tok pisin": "tpi_Latn", "tswana": "tsn_Latn",
    "tsonga": "tso_Latn", "turkmen": "tuk_Latn",
    "tumbuka": "tum_Latn", "turkish": "tur_Latn",
    "twi": "twi_Latn", "central atlas tamazight": "tzm_Tfng",
    "uyghur": "uig_Arab", "ukrainian": "ukr_Cyrl",
    "umbundu": "umb_Latn", "urdu": "urd_Arab",
    "uzbek": "uzn_Latn", "venetian": "vec_Latn",
    "vietnamese": "vie_Latn", "waray": "war_Latn",
    "wolof": "wol_Latn", "xhosa": "xho_Latn",
    "yiddish": "ydd_Hebr", "yoruba": "yor_Latn",
    "yue chinese": "yue_Hant", "zhongwen": "zho_Hans",
    "zho hant": "zho_Hant", "zulu": "zul_Latn",
}

translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="spa_Latn"
    )

class TranslateInput(BaseModel):
    text: str
    language: str

@tool(args_schema=TranslateInput)
def translate_to(text: str, language: str = "spanish") -> str:
    """Translate the given English `text` into the target `language` (like 'urdu', 'french', or 'portuguese').
    If the user has previously asked something to translate, you can use that as context."""
    
    if not text.strip():
        return "⚠️ No text provided to translate. Please include the sentence or phrase you want translated."

    lang_lower = language.strip().lower()
    if lang_lower not in LANG_CODES:
        return f"Unsupported language: '{lang_lower}'"

    tgt = LANG_CODES[lang_lower]
    translator.tokenizer.src_lang = "eng_Latn"
    translator.tokenizer.tgt_lang = tgt
    result = translator(text)

    return result[0]["translation_text"]

# Initialize the chat model

llm = ChatOpenAI(temperature=0 , model='gpt-3.5-turbo' , openai_api_key=OPENAI_API_KEY)

tools = [
    web_search,
    summarize,
    translate_to
]

memory = ConversationBufferMemory(memory_key="chat_history" , return_messages=True)

# Define the prompt template that uses memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can translate, summarize, and web search using tools. \
    If the user asks to 'translate to Urdu' or 'now in French', assume they want to translate their last message."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
# Create the agent with the prompt
agent_chain = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Plug in the memory here
agent_executor = AgentExecutor(
    agent=agent_chain,
    tools=tools,
    memory=memory,
    return_only_outputs=True,
)
