from not_an_llm.analysis.topic_modeling.model import TopicModelingResult, assign_topics

__all__ = [
    "TopicModelingResult",
    "assign_topics",
    "run_topic_analysis",
    "run_topic_modeling",
]


def __getattr__(name: str):
    if name in {"run_topic_analysis", "run_topic_modeling"}:
        from importlib import import_module

        pipeline = import_module("not_an_llm.analysis.topic_modeling.pipeline")
        return getattr(pipeline, name)
    raise AttributeError(name)
