//! Symbolic string-transformation tasks grounded in English instruction words.
//!
//! Each instance teaches ONE operation (e.g. the word "reve" meaning reverse)
//! through in-stream demonstration pairs — `reve cat tac` — then probes with
//! fresh query words (`reve dog` → emit `god`). The brain must (a) remember
//! which English instruction word is active across the whole episode, and
//! (b) execute the bound transformation character-by-character. Name bindings
//! are fixed in v0; the co-evolutionary forge will scramble them later so
//! grounding is earned by demonstration alone.
//!
//! Operations span a difficulty ladder:
//! - `copy`  identity (k=1 causal, memory of one word-span required)
//! - `dupl`  double each character
//! - `rota`  rotate first character to the end (needs neighbor context)
//! - `cyph`  random per-instance substitution cipher (induction from demos)
//! - `reve`  reverse (anti-causal streaming: requires whole-word memory)

use crate::{mix64, Observation, SymbolicTask, Transition};
use anyhow::{bail, Result};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use types::Symbol;

pub const OP_WORDS: [&str; 5] = ["copy", "reve", "rota", "dupl", "cyph"];

const DOMAIN: u64 = 0x5359_4D43_4F4D_5055; // "SYMCOMPU"

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicComputeConfig {
    pub ops: Vec<String>,
    pub word_len: usize,
    pub n_demos: usize,
    pub n_probes: usize,
    pub learning_passes: usize,
    pub predictive_coding: bool,
}

impl Default for SymbolicComputeConfig {
    fn default() -> Self {
        Self {
            ops: OP_WORDS.iter().map(|s| s.to_string()).collect(),
            word_len: 4,
            n_demos: 4,
            n_probes: 2,
            learning_passes: 3,
            predictive_coding: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SymbolicComputeTask {
    pub config: SymbolicComputeConfig,
}

fn op_word(name: &str) -> Vec<Symbol> {
    name.chars().map(|c| Symbol::from_ascii_char(c).expect("op names are ascii")).collect()
}

fn apply_op(op: &str, src: &[char], cipher: &[u8; 26]) -> String {
    match op {
        "copy" => src.iter().collect(),
        "reve" => src.iter().rev().collect(),
        "rota" => {
            let mut out = String::with_capacity(src.len());
            for c in &src[1..] {
                out.push(*c);
            }
            out.push(src[0]);
            out
        }
        "dupl" => src.iter().flat_map(|c| [*c, *c]).collect(),
        "cyph" => src.iter().map(|c| cipher[((*c as u8) - b'a') as usize] as char).collect(),
        other => unreachable!("validated op `{other}`"),
    }
}

pub struct SymbolicComputeState {
    /// Teacher-forced demonstration stream, including END terminators.
    learning: Vec<Symbol>,
    /// One token episode per scored probe.
    probes: Vec<Vec<Symbol>>,
    /// First index within a probe episode where characters are scored.
    scored_start: usize,
    position: usize,
    learning_episode_ok: bool,
    probe_index: usize,
    probe_cursor: usize,
    probe_episode_ok: bool,
    done: bool,
}

impl SymbolicComputeState {
    fn observe_learning(&self) -> Observation {
        let symbol = if self.position == 0 {
            Symbol::End
        } else {
            self.learning[self.position - 1]
        };
        Observation { symbol: Some(symbol) }
    }

    fn observe_probe(&self) -> Observation {
        let stream = &self.probes[self.probe_index];
        let symbol = if self.probe_cursor == 0 {
            Symbol::End
        } else {
            stream[self.probe_cursor - 1]
        };
        Observation { symbol: Some(symbol) }
    }
}

impl SymbolicComputeTask {
    fn build_instance(
        &self,
        rng: &mut ChaCha8Rng,
    ) -> (Vec<Symbol>, Vec<(Vec<Symbol>, usize)>) {
        let op_name = self.config.ops[rng.random_range(0..self.config.ops.len())].clone();
        let mut cipher: Vec<u8> = (b'a'..=b'z').collect();
        cipher.shuffle(rng);
        let mut cipher_arr = [0_u8; 26];
        cipher_arr.copy_from_slice(&cipher);

        let mut word = || -> Vec<char> {
            (0..self.config.word_len).map(|_| (rng.random_range(b'a'..=b'z')) as char).collect()
        };

        let mut learning = Vec::new();
        for _ in 0..self.config.n_demos {
            let src = word();
            let tgt = apply_op(&op_name, &src, &cipher_arr);
            for c in op_word(&op_name) {
                learning.push(c);
            }
            learning.push(Symbol::Space);
            for c in &src {
                learning.push(Symbol::from_ascii_char(*c).expect("lowercase"));
            }
            learning.push(Symbol::Space);
            for c in tgt.chars() {
                learning.push(Symbol::from_ascii_char(c).expect("cipher output is lowercase"));
            }
            learning.push(Symbol::End);
        }

        let mut probes = Vec::new();
        let mut scored_starts = Vec::new();
        for _ in 0..self.config.n_probes {
            let q_src = word();
            let q_tgt = apply_op(&op_name, &q_src, &cipher_arr);
            let mut stream = Vec::new();
            for c in op_word(&op_name) {
                stream.push(c);
            }
            stream.push(Symbol::Space);
            for c in &q_src {
                stream.push(Symbol::from_ascii_char(*c).expect("lowercase"));
            }
            stream.push(Symbol::Space);
            // Scoring region begins at the first target character and includes
            // the trailing END (predicting termination after a full answer).
            let scored_start = stream.len();
            for c in q_tgt.chars() {
                stream.push(Symbol::from_ascii_char(c).expect("lowercase"));
            }
            stream.push(Symbol::End);
            scored_starts.push(scored_start);
            probes.push(stream);
        }

        (learning, probes.into_iter().zip(scored_starts).collect())
    }
}

impl SymbolicTask for SymbolicComputeTask {
    type Config = SymbolicComputeConfig;
    type State = SymbolicComputeState;

    fn name(&self) -> &'static str {
        "symbolic_compute"
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }

    fn validate(&self) -> Result<()> {
        if self.config.ops.is_empty() {
            bail!("symbolic task needs at least one operation");
        }
        for op in &self.config.ops {
            if !OP_WORDS.contains(&op.as_str()) {
                bail!("unknown operation `{op}`; valid: {OP_WORDS:?}");
            }
        }
        if self.config.word_len < 2 {
            bail!("word length must be at least two");
        }
        if self.config.n_demos == 0 || self.config.n_probes == 0 {
            bail!("demos and queries must be positive");
        }
        if self.config.learning_passes == 0 {
            bail!("learning passes must be positive");
        }
        Ok(())
    }

    fn observes_symbols(&self) -> bool {
        true
    }

    fn reveals_teaching_targets(&self) -> bool {
        true
    }

    fn uses_value_critic(&self) -> bool {
        !self.config.predictive_coding
    }

    fn action_enabled(&self, _action: Symbol) -> bool {
        true
    }

    fn max_steps_per_instance(&self) -> usize {
        // Recomputed cheaply: demo episodes dominate; mirrors state size below.
        self.config.n_demos
            * (4 + 1 + self.config.word_len + 1 + self.config.word_len * 2 + 1)
            * self.config.learning_passes
    }

    fn start(&self, panel_seed: u64, instance: usize) -> Self::State {
        let mut rng =
            ChaCha8Rng::seed_from_u64(mix64(panel_seed ^ DOMAIN ^ (instance as u64).rotate_left(23)));
        let (mut learning, probes_raw) = self.build_instance(&mut rng);

        // Duplicate the learning stream once per pass (teacher forcing).
        let single = learning.clone();
        for _ in 1..self.config.learning_passes {
            learning.extend_from_slice(&single);
        }

        let probes = probes_raw.iter().map(|(stream, _)| stream.clone()).collect::<Vec<_>>();
        let scored_start = probes_raw.first().map(|(_, st)| *st).unwrap_or(0);

        SymbolicComputeState {
            learning,
            probes,
            scored_start,
            position: 0,
            learning_episode_ok: true,
            probe_index: 0,
            probe_cursor: 0,
            probe_episode_ok: true,
            done: false,
        }
    }

    fn observe(&self, state: &Self::State) -> Observation {
        state.observe_learning()
    }

    fn step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.learning[state.position];
        let correct = action == expected;
        if correct {
            state.learning_episode_ok &= true;
        } else {
            state.learning_episode_ok = false;
        }
        state.position += 1;
        let trial_outcome =
            if expected == Symbol::End { Some(state.learning_episode_ok) } else { None };
        if expected == Symbol::End {
            state.learning_episode_ok = true;
        }
        Transition {
            reward: if correct { 1.0 } else { -1.0 },
            expected_action: Some(expected),
            teaching_target: Some(expected),
            success_events: 0,
            correct,
            trial_outcome,
            done: state.position >= state.learning.len(),
        }
    }

    fn probe_steps_per_instance(&self) -> usize {
        self.config.n_probes * (5 + self.config.word_len + 1 + self.config.word_len * 2 + 1)
    }

    fn begin_probe(&self, state: &mut Self::State) {
        state.probe_index = 0;
        state.probe_cursor = 0;
        state.probe_episode_ok = true;
        state.done = false;
    }

    fn probe_observe(&self, state: &Self::State) -> Observation {
        state.observe_probe()
    }

    fn probe_step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let stream = &state.probes[state.probe_index];
        let expected = stream[state.probe_cursor];
        let scored = state.probe_cursor >= state.scored_start;
        let correct = action == expected;
        if scored && !correct {
            state.probe_episode_ok = false;
        }
        state.probe_cursor += 1;
        let at_end = expected == Symbol::End;
        let last_probe = state.probe_index + 1 == state.probes.len();
        let success_events = u32::from(scored && correct);
        let trial_outcome = at_end.then_some(state.probe_episode_ok);
        if at_end {
            if last_probe {
                state.done = true;
            } else {
                state.probe_index += 1;
                state.probe_cursor = 0;
                state.probe_episode_ok = true;
            }
        }
        Transition {
            reward: if correct { 1.0 } else { -1.0 },
            expected_action: Some(expected),
            teaching_target: None,
            success_events,
            correct,
            trial_outcome,
            done: state.done,
        }
    }
}
