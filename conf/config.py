import os
from pathlib import Path
import pyprojroot
from dotenv import load_dotenv
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

BASE = pyprojroot.find_root(pyprojroot.has_dir("conf"))
HOME = os.path.expanduser("~")


load_dotenv(Path(BASE, ".env.dev"))


@dataclass
class User:
    bio: str = str(Path(BASE, "data", "raw", "bio.json"))
    facts: str = str(Path(BASE, "data", "raw", "facts.json"))
    events: str = str(Path(BASE, "data", "raw", "events.json"))


@dataclass
class Survey:
    survey_answers: str = str(
        Path(BASE, "data", "raw", "survey", "survey_answers.json")
    )
    survey_scored: str = str(Path(BASE, "data", "raw", "survey", "survey_scored.json"))

    survey_demographics2_questions: str = str(
        Path(BASE, "data", "raw", "survey", "survey_demographics2_questions.json")
    )
    survey_demographics_questions: str = str(
        Path(BASE, "data", "raw", "survey", "survey_demographics_questions.json")
    )
    survey_gad7_questions: str = str(
        Path(BASE, "data", "raw", "survey", "survey_gad7_questions.json")
    )
    survey_phq9_questions: str = str(
        Path(BASE, "data", "raw", "survey", "survey_phq9_questions.json")
    )

    survey_gad7_scoring: str = str(
        Path(BASE, "data", "raw", "survey", "survey_gad7_scoring.json")
    )
    survey_phq9_scoring: str = str(
        Path(BASE, "data", "raw", "survey", "survey_phq9_scoring.json")
    )


@dataclass
class Prompts:
    router: str = str(Path(BASE, "data", "prompts", "llm_router", "router.txt"))

    empathetic_pre_NS: str = str(
        Path(BASE, "data", "prompts", "empathetic", "pre_NS.txt")
    )
    empathetic_during_NS: str = str(
        Path(BASE, "data", "prompts", "empathetic", "during_NS.txt")
    )
    empathetic_post_NS: str = str(
        Path(BASE, "data", "prompts", "empathetic", "post_NS.txt")
    )

    counsellor_human_NS: str = str(
        Path(BASE, "data", "prompts", "counsellor", "human", "NS.txt")
    )
    counsellor_system_pre_NS: str = str(
        Path(BASE, "data", "prompts", "counsellor", "system", "pre_NS.txt")
    )
    counsellor_system_during_NS: str = str(
        Path(BASE, "data", "prompts", "counsellor", "system", "during_NS.txt")
    )
    counsellor_system_post_NS: str = str(
        Path(BASE, "data", "prompts", "counsellor", "system", "post_NS.txt")
    )

    rag_ns_human_NS: str = str(
        Path(BASE, "data", "prompts", "RAG_NS", "human", "NS.txt")
    )
    rag_ns_system_pre_NS: str = str(
        Path(BASE, "data", "prompts", "RAG_NS", "system", "pre_NS.txt")
    )
    rag_ns_system_during_NS: str = str(
        Path(BASE, "data", "prompts", "RAG_NS", "system", "during_NS.txt")
    )
    rag_ns_system_post_NS: str = str(
        Path(BASE, "data", "prompts", "RAG_NS", "system", "post_NS.txt")
    )

    survey_system_during_NS: str = str(
        Path(BASE, "data", "prompts", "survey", "during_NS.txt")
    )

    metadata_gen_desc: str = str(
        Path(BASE, "data", "prompts", "metadata_gen", "descriptions_gen.txt")
    )
    metadata_gen_keywords: str = str(
        Path(BASE, "data", "prompts", "metadata_gen", "keywords_gen.txt")
    )
    metadata_gen_name: str = str(
        Path(BASE, "data", "prompts", "metadata_gen", "name_gen.txt")
    )


@dataclass
class Log:
    log_config: str = str(Path(BASE, "conf", "logging.yaml"))


@dataclass
class Server:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    user: User = field(default_factory=User)
    survey: Survey = field(default_factory=Survey)
    prompts: Prompts = field(default_factory=Prompts)
    log: Log = field(default_factory=Log)
    server: Server = field(default_factory=Server)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cfg = Config()

system_config = {
    "model_type": "openai", #groq/openai
    "model_name": "gpt-4o", #gpt-4o-mini/gpt-4o
    "temperature": 0.1,
    "streaming": False,
    "days_thresholds": {
        "gad7": 2,
        "phq9": 1
    },
    "survey_answers_file_path": cfg.survey.survey_answers,
    "embedding": {"type": "OpenAIEmbeddings", "model": "text-embedding-ada-002"},
    "search_type": "hybrid",
    "top_k": 5,
    "search_weight": 0.5,
    "metadata_filter_NS": {"cat0": "NS_RAG"},
    "metadata_filter_counselor": {"cat0": "counselor_RAG"},
}
