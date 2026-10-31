from app.agents.context_evaluator import requires_current_web_information as needs_web


def test_paper_question_mentioning_news_articles_is_not_current():
    assert not needs_web(
        "In the GPT-3 paper's human-detection experiment, how accurately could people "
        "distinguish GPT-3 175B-generated news articles from human-written ones?"
    )


def test_substrings_inside_other_words_do_not_trigger():
    assert not needs_web("How does concurrent training handle a priceless dataset?")


def test_genuine_current_information_still_triggers():
    assert needs_web("What is the latest GPT release?")
    assert needs_web("Any breaking news on LLM regulation?")
    assert needs_web("What is the pricing today?")
