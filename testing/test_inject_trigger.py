from toolkit.prompt_utils import inject_trigger_into_prompt


def test_inject_trigger_single_prepend():
    prompt = "a portrait"
    trigger = "Jinx"
    out = inject_trigger_into_prompt(prompt, trigger)
    assert out.startswith("Jinx ")


def test_inject_trigger_multi_no_prepend():
    prompt = "a portrait"
    trigger = "Jinx, Zapper"
    out = inject_trigger_into_prompt(prompt, trigger)
    # Should NOT prepend the CSV trigger string
    assert not out.startswith("Jinx, Zapper ")
    # Should remain unchanged except placeholder replacement
    assert out == "a portrait"


def test_inject_trigger_placeholder_with_csv():
    prompt = "[trigger] does something"
    trigger = "Jinx, Zapper"
    out = inject_trigger_into_prompt(prompt, trigger)
    # Placeholder should be replaced by csv string
    assert out.startswith("Jinx, Zapper")
    assert "does something" in out
