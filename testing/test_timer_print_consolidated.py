from toolkit.timer import Timer


def test_print_uses_single_print_call(monkeypatch):
    calls = []

    def fake_print_acc(s):
        calls.append(s)

    import toolkit.print as tprint
    monkeypatch.setattr(tprint, 'print_acc', fake_print_acc)

    t = Timer('test', max_buffer=5)
    # populate a couple timers
    t.start('a')
    t.stop('a')
    t.start('b')
    t.stop('b')

    t.print()

    assert len(calls) == 1
    assert "Timer 'test':" in calls[0]
    assert 'a' in calls[0]
    assert 'b' in calls[0]
