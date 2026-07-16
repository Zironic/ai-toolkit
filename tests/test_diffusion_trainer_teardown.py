from unittest import mock

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer


def test_ui_thread_pool_shutdown_is_idempotent():
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    thread_pool = mock.Mock()
    trainer.thread_pool = thread_pool

    trainer._shutdown_thread_pool()
    trainer._shutdown_thread_pool()

    thread_pool.shutdown.assert_called_once_with(wait=True)
    assert trainer.thread_pool is None
