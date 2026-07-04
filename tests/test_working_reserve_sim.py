"""Invariants for the auto-working_reserve controller, checked via the offline simulator.

These guard the control *dynamics* (converge, hold, no-spill, recover) without a
GPU. The simulator drives the real decision functions
(MemoryManager._available_vram_gib / _training_layout_action), so a regression in
those — or a constant retune that breaks convergence — fails here in milliseconds
instead of after a multi-hour training run.
"""

import importlib.util as u
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = u.spec_from_file_location(
    "sim_working_reserve_controller", ROOT / "scripts" / "sim_working_reserve_controller.py"
)
sim = u.module_from_spec(spec)
# Register before exec: @dataclass resolves field types via sys.modules[__module__].
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)


class WorkingReserveSimTests(unittest.TestCase):
    def _assert_converges_and_holds(self, cfg, history, *, allow_spills=0):
        s = sim.summarize(history, cfg)
        self.assertIsNotNone(s["converged_at"], f"never converged: {s}")
        # The whole point of the hold band: zero layout moves after convergence,
        # so the prefetch trace survives.
        self.assertEqual(s["tail_moves"], 0, f"churned after converging: {s}")
        self.assertTrue(s["settled_in_band"], f"final cycle out of band: {s}")
        self.assertLessEqual(s["spills"], allow_spills, f"too many spills: {s}")
        return s

    def test_single_layer_converges_and_holds(self):
        cfg, history = sim.scenario_default()
        s = self._assert_converges_and_holds(cfg, history)
        # Documents the cost of the current single-layer cadence: slow warmup.
        self.assertGreater(s["promotes"], 40)

    def test_batch_converges_far_faster_than_single(self):
        cfg_s, hist_s = sim.scenario_default()
        cfg_b, hist_b = sim.scenario_batch()
        self._assert_converges_and_holds(cfg_b, hist_b)
        single = sim.summarize(hist_s, cfg_s)
        batch = sim.summarize(hist_b, cfg_b)
        # Batch promotion = far fewer layout changes (= far fewer trace resets)
        # and far earlier convergence.
        self.assertLess(batch["promotes"], single["promotes"])
        self.assertLess(batch["converged_at"], single["converged_at"])

    def test_no_spill_in_steady_state(self):
        cfg, history = sim.scenario_batch()
        self.assertEqual(sim.summarize(history, cfg)["spills"], 0)

    def test_small_card_still_holds_without_spill(self):
        cfg, history = sim.scenario_small_card()
        self._assert_converges_and_holds(cfg, history)

    def test_timing_spill_sim_learns_wddm_floor_and_retreats(self):
        cfg, history = sim.scenario_timing_spill_learning()
        timing_events = [r for r in history if r.move == "timing_spill"]
        self.assertGreater(len(timing_events), 0, "sim never hit the timing cliff")
        learned_floor = max(r.governing for r in history)
        self.assertGreater(learned_floor, cfg.wddm_hard_gib)
        first_event_step = timing_events[0].step
        self.assertTrue(
            any(r.step > first_event_step and r.move == "down" for r in history),
            "learned timing cliff did not trigger a retreat",
        )
        final_cycle = history[-4:]
        self.assertTrue(
            all(r.available >= learned_floor - 1e-6 for r in final_cycle),
            f"final cycle fell below learned floor {learned_floor}: {final_cycle}",
        )
    def test_recovers_from_external_pressure(self):
        cfg, history = sim.scenario_pressure()
        s = sim.summarize(history, cfg)
        # A sudden external app at the ceiling may briefly spill, but the
        # controller must demote, then re-promote once it leaves, and end held.
        self.assertGreater(s["demotes"], 0, f"never demoted under pressure: {s}")
        self.assertTrue(s["settled_in_band"], f"did not re-settle: {s}")
        self.assertEqual(s["tail_moves"], 0, f"churning after recovery: {s}")

    def test_noise_does_not_cause_steady_state_spills(self):
        # Under realistic per-step noise the controller must never settle so close
        # to the ceiling that ordinary activation jitter spills.
        cfg, history = sim.scenario_noisy_batch()
        s = sim.summarize(history, cfg)
        self.assertEqual(s["steady_spills"], 0, f"noise spilled in steady state: {s}")
        self.assertTrue(s["settled_in_band"], f"noise knocked it out of band: {s}")

    def test_smoothing_reduces_noise_churn(self):
        # The deterministic sim hid it, but instantaneous governing churns under
        # noise (every move = a trace reset). EMA smoothing must reduce that.
        cfg_raw, hist_raw = sim.scenario_noisy_batch()
        cfg_sm, hist_sm = sim.scenario_noisy_smoothed()
        raw = sim.summarize(hist_raw, cfg_raw)
        smooth = sim.summarize(hist_sm, cfg_sm)
        self.assertLessEqual(smooth["steady_moves"], raw["steady_moves"])
        self.assertEqual(smooth["steady_spills"], 0)

    # --- manual working_reserve + cliff guard --------------------------------

    def test_manual_mode_without_guard_blows_up(self):
        # Sanity that the simulator actually reproduces the observed bug: with the
        # guard OFF, allocator fragmentation ratchets reserved past the ceiling and
        # the run spills on essentially every step (this is what manual mode did
        # before the fix). If this stops failing, the scenario no longer models the
        # failure and the guard test below proves nothing.
        cfg, history = sim.scenario_manual_no_guard()
        s = sim.summarize_manual(history, cfg)
        self.assertGreater(s["steady_spills"], 0, f"no-guard did not spill: {s}")
        self.assertGreater(
            s["max_reserved"], cfg.total_gib,
            f"reserved never ratcheted past the ceiling: {s}",
        )

    def test_manual_guard_prevents_spill(self):
        # Same fragmentation pressure, guard ON: it must reclaim the idle allocator
        # cache and keep driver-free at/above the hard floor with zero spills.
        cfg, history = sim.scenario_manual_cliff_guard()
        s = sim.summarize_manual(history, cfg)
        self.assertEqual(s["spills"], 0, f"guard let it spill: {s}")
        self.assertGreater(s["empty_caches"], 0, f"guard never reclaimed: {s}")
        self.assertLess(
            s["max_reserved"], cfg.total_gib,
            f"reserved blew past the ceiling despite guard: {s}",
        )
        self.assertGreaterEqual(
            round(s["min_free"], 2), cfg.wddm_hard_gib,
            f"guard settled below the hard floor: {s}",
        )

    def test_manual_guard_demotes_when_empty_cache_insufficient(self):
        # When the LIVE footprint alone sits near the ceiling, returning idle cache
        # is not enough — the guard must escalate and demote resident layers. It
        # still must not spill in steady state, and the resident set must shrink.
        cfg, history = sim.scenario_manual_overcommit()
        s = sim.summarize_manual(history, cfg)
        self.assertGreater(s["demotes"], 0, f"guard never demoted: {s}")
        self.assertEqual(s["steady_spills"], 0, f"overcommit spilled in steady state: {s}")
        self.assertLess(
            s["final_resident"], cfg.start_resident,
            f"resident set did not shrink under over-commit: {s}",
        )

    def test_manual_guard_demotes_before_known_shape_under_pressure(self):
        cfg = sim._manual_cfg(
            always_resident_gib=2.6,
            start_resident=30,
            working_floor_gib={"res512": 5.0},
            frag_gib=0.0,
            frag_ratchet_gib=0.0,
            manual_residual_gib=0.2,
        )

        def external(step):
            return 0.8 if step >= 1 else 0.0

        history = sim.run_manual_sim(
            cfg, ["res512"], steps=3, guard=True, external_gib=external
        )
        s = sim.summarize_manual(history, cfg)
        self.assertEqual(history[0].move, "hold")
        self.assertEqual(history[1].move, "pre_down")
        self.assertGreater(s["pre_demotes"], 0, f"known shape did not demote before step: {s}")
        self.assertEqual(s["spills"], 0, f"pre-step demotion failed to avoid spill: {s}")

    def test_manual_guard_demotes_when_post_trim_trough_hides_peak(self):
        cfg = sim._manual_cfg(
            always_resident_gib=2.6,
            start_resident=40,
            working_floor_gib={"res768": 5.6, "res512": 5.0, "res256": 4.6},
            manual_residual_gib=0.2,
        )
        history = sim.run_manual_sim(
            cfg, ["res512", "res256", "res768", "res512"], steps=20, guard=True
        )
        s = sim.summarize_manual(history, cfg)
        self.assertGreater(s["demotes"], 0, f"guard trusted post-trim trough: {s}")
        self.assertEqual(s["steady_spills"], 0, f"peak-pressure demotion did not stabilize: {s}")
        first = history[0]
        self.assertEqual(first.move, "down")
        self.assertGreater(first.governing, cfg.wddm_hard_gib)

    # --- working_reserve SIZING controller (auto) ----------------------------

    def test_reserve_peak_fix_converges_and_holds(self):
        # With the truthful peak signal + cross-bucket governance, the reserve
        # climbs down to ~peak+pad and holds there with a healthy free margin and
        # no retreats — i.e. auto working_reserve actually works.
        cfg, history = sim.scenario_reserve_peak_fix()
        s = sim.summarize_reserve(history, cfg)
        peak = max(cfg.act_peak_gib.values())
        self.assertEqual(s["steady_spills"], 0, f"spilled: {s}")
        self.assertEqual(s["retreats"], 0, f"thrashed (retreated) despite fix: {s}")
        self.assertEqual(s["tail_moves"], 0, f"never settled: {s}")
        self.assertGreaterEqual(
            s["min_free"], cfg.wddm_hard_gib, f"entered the WDDM danger zone: {s}"
        )
        self.assertGreaterEqual(s["final_reserve"], peak, f"reserve under the peak: {s}")
        self.assertLessEqual(
            s["final_reserve"], peak + cfg.pad_gib + cfg.step_gib,
            f"reserve did not converge near peak+pad: {s}",
        )

    def test_reserve_low_seed_climbs_to_peak_and_holds(self):
        # The live auto bug: the reserve is SEEDED below the real activation peak
        # (~2.5 vs ~5.2). The grow branch must climb it straight up to ~peak+pad
        # and hold there with no steady-state spills -- "respect the working set".
        # Without the grow branch the shrink-only decision can only crawl up via
        # hard-floor retreats, spilling on the way and settling under the peak.
        cfg, history = sim.scenario_reserve_low_seed()
        s = sim.summarize_reserve(history, cfg)
        peak = max(cfg.act_peak_gib.values())
        self.assertLess(cfg.start_reserve_gib, peak, "scenario must seed below the peak")
        self.assertGreaterEqual(
            s["final_reserve"], peak, f"reserve never reached the activation peak: {s}"
        )
        self.assertLessEqual(
            s["final_reserve"], peak + cfg.pad_gib + cfg.step_gib,
            f"reserve overshot the activation peak: {s}",
        )
        self.assertEqual(s["steady_spills"], 0, f"still spilling after climb: {s}")
        self.assertEqual(s["tail_moves"], 0, f"never settled after climbing: {s}")
        self.assertGreaterEqual(
            s["min_free"], 0.0, f"went underwater (negative free) at some point: {s}"
        )

    def test_reserve_trough_bug_is_worse_than_peak_fix(self):
        # The old behaviour: sizing off the step-end residual (trough). It cannot
        # find the efficient stable point: it undersizes reserve and rides the
        # cliff. Reserve no longer reacts to the cliff itself; layout safety owns
        # that. So the failure is visible as steady spills plus too-low reserve.
        _, hist_bug = sim.scenario_reserve_trough_bug()
        cfg_fix, hist_fix = sim.scenario_reserve_peak_fix()
        bug = sim.summarize_reserve(hist_bug, cfg_fix)
        fix = sim.summarize_reserve(hist_fix, cfg_fix)
        self.assertLess(bug["min_free"], fix["min_free"], f"bug not nearer cliff: {bug} vs {fix}")
        self.assertLess(bug["min_free"], cfg_fix.wddm_hard_gib,
                        f"bug should breach the danger zone: {bug}")
        self.assertGreater(bug["steady_spills"], fix["steady_spills"],
                           f"bug should keep spilling versus the fix: {bug} vs {fix}")
        self.assertEqual(bug["retreats"], 0,
                         f"working reserve should not panic-retreat on cliffs: {bug}")
        self.assertLess(bug["final_reserve"], fix["final_reserve"],
                        f"bug should stay undersized versus the peak fix: {bug} vs {fix}")

    def test_reserve_cross_bucket_prevents_high_res_starvation(self):
        # Peak signal but per-bucket (no cross-bucket governance): a quiet low-res
        # step can shrink the reserve below what the next high-res step needs.
        # Cross-bucket governance keeps the high-res peak binding even when the
        # current low-res bucket is quiet.
        _, hist_pb = sim.scenario_reserve_per_bucket()
        cfg, hist_fix = sim.scenario_reserve_peak_fix()
        pb = sim.summarize_reserve(hist_pb, cfg)
        fix = sim.summarize_reserve(hist_fix, cfg)
        self.assertLess(pb["min_free"], fix["min_free"], f"per-bucket not worse: {pb} vs {fix}")
        self.assertLess(pb["final_reserve"], fix["final_reserve"],
                        f"per-bucket reserve should be lower than the cross-bucket fix: {pb} vs {fix}")
        self.assertEqual(fix["retreats"], 0, f"cross-bucket should not retreat: {fix}")

    def test_reserve_ignores_external_pressure(self):
        # A transient external app eats the free margin, but that is not part of
        # the activation working set. Layout demotion handles pressure; reserve
        # should stay sized to the measured peak and must not panic-retreat.
        cfg, history = sim.scenario_reserve_pressure()
        s = sim.summarize_reserve(history, cfg)
        peak = max(cfg.act_peak_gib.values()) + cfg.pad_gib
        self.assertEqual(s["retreats"], 0, f"reserve reacted to pressure: {s}")
        self.assertAlmostEqual(s["final_reserve"], peak, places=6)

    def test_auto_seed_defaults_to_five_gib(self):
        from toolkit.memory_management import MemoryManager

        self.assertAlmostEqual(
            MemoryManager._training_auto_seed_working_reserve_gib(),
            5.0,
            places=6,
        )

    def test_panic_reserve_snaps_to_measured_target(self):
        from toolkit.memory_management import MemoryManager

        new_reserve, danger, action = MemoryManager._training_working_reserve_decision(
            current_gib=17.0,
            measured_peak_gib=4.38,
            min_device_free_gib=3.86,
            danger_gib=16.0,
            wddm_hard_gib=1.0,
            wddm_stop_gib=2.0,
            pad_gib=0.5,
            step_gib=0.5,
            retreat_gib=1.0,
        )
        self.assertEqual(action, "shrink")
        self.assertAlmostEqual(new_reserve, 4.88, places=6)
        self.assertIsNone(danger)

    def test_cliff_does_not_change_measured_reserve_target(self):
        from toolkit.memory_management import MemoryManager

        new_reserve, danger, action = MemoryManager._training_working_reserve_decision(
            current_gib=17.0,
            measured_peak_gib=4.38,
            min_device_free_gib=0.0,
            danger_gib=16.0,
            wddm_hard_gib=1.0,
            wddm_stop_gib=2.0,
            pad_gib=0.5,
            step_gib=0.5,
            retreat_gib=1.0,
        )
        self.assertEqual(action, "shrink")
        self.assertAlmostEqual(new_reserve, 4.88, places=6)
        self.assertIsNone(danger)

    def test_observed_driver_free_overrides_optimistic_estimate(self):
        from toolkit.memory_management import MemoryManager

        self.assertAlmostEqual(
            MemoryManager._training_governing_free_gib(
                3.4, {"device_peak_source": "observed", "device_free_peak_gb": 0.0}
            ),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            MemoryManager._training_governing_free_gib(
                3.4, {"device_peak_source": "estimate", "device_free_peak_gb": 0.0}
            ),
            3.4,
            places=6,
        )

    def test_underreserved_peak_grows_before_hard_floor(self):
        from toolkit.memory_management import MemoryManager

        new_reserve, danger, action = MemoryManager._training_working_reserve_decision(
            current_gib=2.03,
            measured_peak_gib=3.39,
            min_device_free_gib=1.20,
            danger_gib=None,
            wddm_hard_gib=1.0,
            wddm_stop_gib=1.5,
            pad_gib=0.5,
            step_gib=0.5,
        )
        self.assertEqual(action, "grow")
        self.assertIsNone(danger)
        self.assertAlmostEqual(new_reserve, 3.89, places=6)

    def test_working_reserve_signal_tracks_peak_not_floor(self):
        from toolkit.memory_management import MemoryManager
        sig = MemoryManager._training_working_reserve_signal
        # Warmed up: even with a stale-low EMA, the live peak governs the signal.
        self.assertAlmostEqual(
            sig(4.46, 1.1, steps=5, stable_windows=2,
                min_working_reserve_gib=1.5, pad_gib=0.5),
            4.46, places=6,
        )
        # Before warmup: the raw peak is used directly.
        self.assertAlmostEqual(
            sig(4.46, None, steps=1, stable_windows=2,
                min_working_reserve_gib=1.5, pad_gib=0.5),
            4.46, places=6,
        )
        # Floor: a quiet bucket cannot drag the signal below min_working_reserve-pad.
        self.assertAlmostEqual(
            sig(0.2, 0.2, steps=5, stable_windows=2,
                min_working_reserve_gib=1.5, pad_gib=0.5),
            1.0, places=6,
        )

    def test_sampling_guard_predicts_peak_free_and_external_pressure(self):
        # Reproduces the asleep-vs-awake batched-CFG run: same plan, only external
        # VRAM grows. The predictor must drop below the guard margin when Windows
        # takes more, so the per-image guard demotes a block.
        from toolkit.memory_management import MemoryManager
        gib = 1024 ** 3
        f = MemoryManager._sampling_guard_predicted_peak_free
        total = int(12.0 * gib)
        peak_reserved = int(7.76 * gib)  # worst forward's reserved high-water
        margin = int(0.5 * gib)          # reactive, tight margin
        # Asleep: free 3.02, reserved 7.76 -> other ~1.21 -> predicted ~0.28.
        asleep = f(total, int(3.02 * gib), int(7.76 * gib), peak_reserved)
        # Awake: Windows took enough more VRAM to push predicted peak free below the guard.
        awake = f(total, int(0.40 * gib), int(7.76 * gib), peak_reserved)
        self.assertGreater(asleep, awake, "external growth must lower predicted peak free")
        self.assertLess(awake, margin, "awake case must trip the guard")
        # A quiet machine (lots of free) must NOT trip the guard.
        quiet = f(total, int(5.0 * gib), int(6.0 * gib), int(6.0 * gib))
        self.assertGreater(quiet, margin, "quiet machine should not demote")

    def test_sampling_step_trim_escalation(self):
        # Per-step trim collapses the streaming-churn reserved high-water before it
        # silently crosses the WDDM cliff. Trigger on REALIZED free (paging is on
        # the committed footprint), trim first, demote only if no idle cache left.
        from toolkit.memory_management import MemoryManager
        gib = 1024 ** 3
        should_trim = MemoryManager._sampling_step_should_trim
        should_demote = MemoryManager._sampling_step_should_demote
        trim_margin = int(1.5 * gib)
        hard_floor = int(1.0 * gib)
        # Plenty of free -> no-op (the common, slack case).
        self.assertFalse(should_trim(int(4.0 * gib), trim_margin))
        # Free dropped into the margin -> trim the cache.
        self.assertTrue(should_trim(int(1.2 * gib), trim_margin))
        # Trim reclaimed idle cache back above the floor -> done, no demote.
        self.assertFalse(should_demote(int(2.5 * gib), hard_floor))
        # Trim reclaimed nothing (genuine external pressure) -> escalate to demote.
        self.assertTrue(should_demote(int(0.6 * gib), hard_floor))

    def test_timing_spill_learns_floor_near_memory_cliff(self):
        from toolkit.memory_management import MemoryManager

        floor = MemoryManager._training_timing_spill_floor(
            31.0,
            10.0,
            0.8,
            steps=8,
            slowdown_ratio=3.0,
            max_signal_free_gib=1.5,
            pad_gib=0.25,
        )
        self.assertAlmostEqual(floor, 1.05, places=6)

    def test_timing_spill_ignores_slowdown_with_plenty_of_free_memory(self):
        from toolkit.memory_management import MemoryManager

        floor = MemoryManager._training_timing_spill_floor(
            31.0,
            10.0,
            4.0,
            steps=8,
            slowdown_ratio=3.0,
            max_signal_free_gib=1.5,
            pad_gib=0.25,
        )
        self.assertIsNone(floor)
    def test_cliff_guard_predicts_next_peak_not_post_trim_trough(self):
        from toolkit.memory_management import MemoryManager

        pred = MemoryManager._training_cliff_predicted_peak_free_gib(
            total_gib=12.0,
            device_free_gib=0.0,
            torch_reserved_gib=11.5,
            peak_allocated_gib=11.2,
        )
        self.assertAlmostEqual(pred, 0.3, places=6)
        healthy = MemoryManager._training_cliff_predicted_peak_free_gib(
            total_gib=12.0,
            device_free_gib=0.0,
            torch_reserved_gib=11.5,
            peak_allocated_gib=5.2,
        )
        self.assertAlmostEqual(healthy, 6.3, places=6)

    def test_cliff_guard_action_predicate(self):
        from toolkit.memory_management import MemoryManager
        act = MemoryManager._training_cliff_guard_action
        # Above the floor and no OOM -> leave manual mode alone.
        self.assertEqual(act(2.0, wddm_hard_gib=1.0), "ok")
        # Below the floor -> reclaim.
        self.assertEqual(act(0.5, wddm_hard_gib=1.0), "reclaim")
        # An OOM forces a reclaim even if the post-OOM free reads healthy.
        self.assertEqual(act(5.0, wddm_hard_gib=1.0, did_oom=True), "reclaim")

    def test_available_vram_accounts_for_other_residents(self):
        from toolkit.memory_management import MemoryManager
        # total=16, reserved=9, other = used - reserved = 12 - 9 = 3, safety=0.5
        # available = (16 - 0.5 - 3) - 9 = 3.5
        avail = MemoryManager._available_vram_gib(16.0, 12.0, 9.0, 9.0, safety_gib=0.5)
        self.assertAlmostEqual(avail, 3.5, places=6)
        # More "other" (e.g. a browser) shrinks available one-for-one.
        avail2 = MemoryManager._available_vram_gib(16.0, 14.0, 9.0, 9.0, safety_gib=0.5)
        self.assertAlmostEqual(avail2, 1.5, places=6)

    def test_provisional_trace_does_not_authorize_resident_growth(self):
        from toolkit.memory_management import MemoryManager

        gate = MemoryManager._prefetch_allows_resident_growth
        self.assertTrue(
            gate(pool_present=False, schedule_confidence="cold", prefetch_healthy=False)
        )
        self.assertTrue(
            gate(pool_present=True, schedule_confidence="exact", prefetch_healthy=True)
        )
        self.assertTrue(
            gate(pool_present=True, schedule_confidence="observed", prefetch_healthy=True)
        )
        self.assertFalse(
            gate(pool_present=True, schedule_confidence="compatible", prefetch_healthy=True)
        )
        self.assertFalse(
            gate(pool_present=True, schedule_confidence="cold", prefetch_healthy=True)
        )
        self.assertFalse(
            gate(pool_present=True, schedule_confidence="exact", prefetch_healthy=False)
        )

    def test_prefetch_trace_invalid_uses_semantic_mismatch_rates(self):
        from toolkit.memory_management import MemoryManager

        invalid = MemoryManager._prefetch_trace_invalid
        self.assertFalse(
            invalid(
                schedule_len=10,
                consume_pos=10,
                lookahead=4,
                hard_miss_rate=0.0,
                mismatch_rate=0.0,
                duplicate_key_block_rate=0.0,
            )
        )
        self.assertTrue(
            invalid(
                schedule_len=10,
                consume_pos=20,
                lookahead=4,
                hard_miss_rate=0.5,
                mismatch_rate=0.0,
                duplicate_key_block_rate=0.0,
            )
        )
        self.assertTrue(
            invalid(
                schedule_len=10,
                consume_pos=10,
                lookahead=4,
                hard_miss_rate=0.0,
                mismatch_rate=0.2,
                duplicate_key_block_rate=0.0,
            )
        )
        self.assertTrue(
            invalid(
                schedule_len=10,
                consume_pos=10,
                lookahead=4,
                hard_miss_rate=0.0,
                mismatch_rate=0.0,
                duplicate_key_block_rate=0.2,
            )
        )

    def test_prefetch_recovery_action_prioritizes_invalid_trace(self):
        from toolkit.memory_management import MemoryManager

        action = MemoryManager._prefetch_recovery_action
        self.assertIsNone(action(prefetch_missing=False, prefetch_invalid=False))
        self.assertEqual(
            action(prefetch_missing=True, prefetch_invalid=False),
            "seed_prefetch_schedule",
        )
        self.assertEqual(
            action(prefetch_missing=False, prefetch_invalid=True),
            "reset_prefetch_trace",
        )
        self.assertEqual(
            action(prefetch_missing=True, prefetch_invalid=True),
            "reset_prefetch_trace",
        )

if __name__ == "__main__":
    unittest.main()
