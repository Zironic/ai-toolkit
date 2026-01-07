from toolkit.timer import Timer


def test_timer_context_class_reused():
    t = Timer('test', max_buffer=5)
    ctx1 = t('a')
    ctx2 = t('a')
    # The type should be identical (class not recreated each call)
    assert type(ctx1) is type(ctx2)

    # ensure basic usage still works
    with t('a'):
        pass
    assert 'a' in t.timers
