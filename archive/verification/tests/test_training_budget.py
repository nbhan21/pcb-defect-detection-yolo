from types import SimpleNamespace

from pcb_inspector.training import TrainingBudget


def test_epoch_limit_wins_even_with_time_remaining():
    budget = TrainingBudget(2, 3600, clock=lambda: 10)
    trainer = SimpleNamespace(epoch=1, stop=False)
    budget.start(trainer)
    budget.check(trainer)
    assert trainer.stop


def test_time_limit_wins_before_last_epoch():
    now = [0.0]
    budget = TrainingBudget(40, 3600, clock=lambda: now[0])
    trainer = SimpleNamespace(epoch=0, stop=False)
    budget.start(trainer)
    now[0] = 3599
    budget.check(trainer)
    assert not trainer.stop
    now[0] = 3601
    budget.check(trainer)
    assert trainer.stop


def test_budget_preserves_framework_early_stop():
    budget = TrainingBudget(40, 3600, clock=lambda: 0)
    trainer = SimpleNamespace(epoch=0, stop=True)
    budget.start(trainer)
    budget.check(trainer)
    assert trainer.stop
