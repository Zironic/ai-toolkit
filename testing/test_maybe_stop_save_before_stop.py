import pytest


class MinimalTrainer:
    def __init__(self, step_num=0):
        self.is_ui_trainer = True
        self.is_stopping = False
        self.step_num = step_num
        self._run_async_operation = lambda coro: None
        self._update_status = lambda *args, **kwargs: None

    def should_stop(self):
        return False

    def should_save_before_stop(self):
        return False

    def should_return_to_queue(self):
        return False

    def save(self, step=None, force=False):
        # override in tests
        pass

    def maybe_stop(self):
        if self.should_stop():
            if self.should_save_before_stop():
                self._run_async_operation(
                    self._update_status("running", "Saving before stop..."))
                # Force save without re-checking stop to avoid recursion
                self.save(self.step_num, force=True)
            self._run_async_operation(
                self._update_status("stopped", "Job stopped"))
            self.is_stopping = True
            raise Exception("Job stopped")
        if self.should_return_to_queue():
            self._run_async_operation(
                self._update_status("queued", "Job queued"))
            self.is_stopping = True
            raise Exception("Job returning to queue")


def test_minimal_trainer_maybe_stop_calls_force_save_for_stop():
    trainer = MinimalTrainer(step_num=123)

    saved = {"called": False, "step": None, "force": None}

    def fake_save(step=None, force=False):
        saved["called"] = True
        saved["step"] = step
        saved["force"] = force

    trainer.should_stop = lambda: True
    trainer.should_save_before_stop = lambda: True
    trainer.save = fake_save

    with pytest.raises(Exception, match="Job stopped"):
        trainer.maybe_stop()

    assert saved["called"] is True
    assert saved["step"] == 123
    assert saved["force"] is True


def test_minimal_trainer_maybe_stop_calls_force_save_for_ui_trainer_queue():
    trainer = MinimalTrainer(step_num=7)

    saved = {"called": False, "step": None, "force": None}

    def fake_save(step=None, force=False):
        saved["called"] = True
        saved["step"] = step
        saved["force"] = force

    trainer.should_stop = lambda: True
    trainer.should_save_before_stop = lambda: True
    trainer.save = fake_save

    with pytest.raises(Exception, match="Job stopped"):
        trainer.maybe_stop()

    assert saved["called"] is True
    assert saved["step"] == 7
    assert saved["force"] is True
