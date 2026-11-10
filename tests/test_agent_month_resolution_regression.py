from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_local_openai_tools_agent.py"

spec = spec_from_file_location("run_local_openai_tools_agent_under_test", SCRIPT_PATH)
module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_complete_iso_date_is_not_treated_as_sales_month():
    assert module.extract_month_from_text("2026-11-09") is None
    assert module.extract_month_from_text("última observação em 2026-11-09") is None


def test_explicit_year_month_is_still_recognized():
    assert module.extract_month_from_text("2026-09") == "2026-09"
    assert module.extract_month_from_text("setembro de 2026") == "2026-09"


def test_unrelated_conceptual_question_does_not_inherit_old_month():
    history = [
        SimpleNamespace(type="human", content="Qual foi a receita líquida em setembro de 2026?"),
        SimpleNamespace(type="ai", content="A meta Selic observada em 2026-11-09 foi 13,75% a.a."),
        SimpleNamespace(type="human", content="Qual é a meta Selic mais recente disponível?"),
    ]
    question = "Explique como é calculada a receita líquida e quais campos são usados."
    assert module.resolve_month(question, history) is None


def test_follow_up_explicitly_referencing_same_period_inherits_month():
    history = [
        SimpleNamespace(type="human", content="Qual foi a receita líquida em setembro de 2026?"),
    ]
    question = "E qual foi a receita bruta desse mesmo período?"
    assert module.resolve_month(question, history) == "2026-09"


def test_explicit_month_in_current_question_takes_precedence():
    history = [
        SimpleNamespace(type="human", content="Qual foi a receita líquida em agosto de 2026?"),
    ]
    question = "Qual foi a receita líquida em setembro de 2026?"
    assert module.resolve_month(question, history) == "2026-09"