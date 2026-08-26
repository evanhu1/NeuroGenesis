use crate::{Observation, SymbolicTask, Transition};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use types::Symbol;

pub const DEFAULT_SNIPPET: &str = "the quick brown fox jumps over the lazy dog";

/// Word pool for generated snippets. Content only — the environment's text
/// distribution, not a representation or strategy. Collectively covers every
/// letter so generated text exercises the whole alphabet.
pub const LEXICON: [&str; 64] = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "walks", "through", "a",
    "quiet", "village", "where", "people", "bake", "bread", "read", "books", "play", "music",
    "watch", "evening", "sky", "river", "flows", "past", "green", "fields", "birds", "sing", "in",
    "old", "trees", "children", "run", "home", "before", "dark", "wind", "moves", "slow", "clouds",
    "above", "quiet", "hills", "farmers", "gather", "wheat", "under", "warm", "light", "boats",
    "drift", "near", "docks", "with", "extra", "boxes", "of", "frozen", "fish", "jazz",
];

const GENERATED_SNIPPET_DOMAIN: u64 = 0x4e58_5447_454e_5f53;

fn default_snippet_length() -> usize {
    48
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextTokenPredictionConfig {
    pub snippet: String,
    pub learning_passes: usize,
    /// Fully-unsupervised predictive-coding mode: no critic, no reward, and each
    /// correct next-character prediction during the stream is one reproductive
    /// ticket (continuous minimal-criterion fitness). The error remains the
    /// prediction-vs-next-observation mismatch, computed by the learner.
    #[serde(default)]
    pub predictive_coding: bool,
    /// Draw a freshly generated snippet for every panel instance instead of
    /// reusing one fixed string. Training, development, and sealed panels are
    /// built from different panel seeds, so sealed snippets are never seen
    /// during evolution: a genome cannot pass by having memorized one
    /// sequence, only by carrying an architecture that acquires an arbitrary
    /// sequence within its own lifetime.
    #[serde(default)]
    pub generalize: bool,
    /// Exact character length of a generated snippet. Fixed so step and probe
    /// budgets stay well defined across instances.
    #[serde(default = "default_snippet_length")]
    pub snippet_length: usize,
    /// Co-evolutionary forge parameters: per-word sampling log-weights over
    /// `LEXICON` (empty = uniform) and a probability of reusing an earlier
    /// word (long-range repetition, a direct memory demand). These are set by
    /// the adversarial forge population in `evolution::co_ecology` runs;
    /// development/sealed audits always evaluate the neutral generator.
    #[serde(default)]
    pub lexicon_bias: Vec<f32>,
    #[serde(default)]
    pub repeat_rate: f32,
}

impl Default for NextTokenPredictionConfig {
    fn default() -> Self {
        Self {
            snippet: DEFAULT_SNIPPET.to_owned(),
            learning_passes: 4,
            predictive_coding: false,
            generalize: false,
            snippet_length: default_snippet_length(),
            lexicon_bias: Vec::new(),
            repeat_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NextTokenPredictionTask {
    pub config: NextTokenPredictionConfig,
}

pub struct NextTokenPredictionState {
    targets: Vec<Symbol>,
    position: usize,
    learning_pass: usize,
    pass_exact: bool,
    probe_position: usize,
    probe_exact: bool,
}

impl NextTokenPredictionTask {
    fn targets(&self) -> Result<Vec<Symbol>> {
        let mut targets = self
            .config
            .snippet
            .chars()
            .map(|character| {
                Symbol::from_ascii_char(character).ok_or_else(|| {
                    anyhow::anyhow!(
                        "next-token snippet accepts only lowercase ASCII letters and spaces"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        targets.push(Symbol::End);
        Ok(targets)
    }

    /// Deterministic per-instance snippet: sample words from the lexicon until
    /// the exact configured length is reached, under the optional forge bias
    /// (per-word log-weights plus a long-range repetition rate). Distinct
    /// panel seeds therefore yield disjoint training and sealed text.
    fn generated_targets(&self, panel_seed: u64, instance: usize) -> Vec<Symbol> {
        let mut state =
            mix64(panel_seed ^ GENERATED_SNIPPET_DOMAIN ^ (instance as u64).rotate_left(23));
        let biased = !self.config.lexicon_bias.is_empty();
        let weights = if biased {
            let max = self
                .config
                .lexicon_bias
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut weights = [0.0_f32; LEXICON.len()];
            let mut total = 0.0;
            for (weight, log_weight) in weights.iter_mut().zip(&self.config.lexicon_bias) {
                *weight = (*log_weight - max).exp();
                total += *weight;
            }
            for weight in &mut weights {
                *weight /= total;
            }
            Some(weights)
        } else {
            None
        };
        let mut used: Vec<&'static str> = Vec::new();
        let mut text = String::with_capacity(self.config.snippet_length + 16);
        while text.chars().count() < self.config.snippet_length {
            if !text.is_empty() {
                text.push(' ');
            }
            state = mix64(state);
            let word = if biased {
                let repeat_roll = mix64(state ^ 0x524550_454154 >> 8) as f32
                    / ((1_u32 << 24) as f32);
                if !used.is_empty()
                    && self.config.repeat_rate > 0.0
                    && repeat_roll < self.config.repeat_rate
                {
                    let pick = (mix64(state ^ 0x5049434b) as usize) % used.len();
                    used[pick]
                } else {
                    let weights = weights.expect("biased mode carries a weight table");
                    let draw = ((state >> 40) as f32 + 0.5) / ((1_u32 << 24) as f32);
                    let mut cumulative = 0.0;
                    let mut chosen = LEXICON[LEXICON.len() - 1];
                    for (lexeme, weight) in LEXICON.iter().zip(&weights) {
                        cumulative += *weight;
                        if draw < cumulative {
                            chosen = lexeme;
                            break;
                        }
                    }
                    chosen
                }
            } else {
                LEXICON[(state % LEXICON.len() as u64) as usize]
            };
            used.push(word);
            text.push_str(word);
        }
        let mut targets = text
            .chars()
            .take(self.config.snippet_length)
            .map(|character| {
                Symbol::from_ascii_char(character).expect("lexicon is lowercase ascii and spaces")
            })
            .collect::<Vec<_>>();
        targets.push(Symbol::End);
        targets
    }

    fn snippet_char_count(&self) -> usize {
        if self.config.generalize {
            self.config.snippet_length
        } else {
            self.config.snippet.chars().count()
        }
    }
}

impl SymbolicTask for NextTokenPredictionTask {
    type Config = NextTokenPredictionConfig;
    type State = NextTokenPredictionState;

    fn name(&self) -> &'static str {
        "basic_next_token_prediction"
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }

    fn validate(&self) -> Result<()> {
        if self.config.generalize {
            if self.config.snippet_length < 8 {
                bail!("generated next-token snippets must be at least eight characters");
            }
            if self.config.learning_passes == 0 {
                bail!("next-token learning passes must be positive");
            }
            if !self.config.lexicon_bias.is_empty() {
                if self.config.lexicon_bias.len() != LEXICON.len() {
                    bail!("lexicon bias must have one entry per lexicon word");
                }
                if self.config.lexicon_bias.iter().any(|w| !w.is_finite()) {
                    bail!("lexicon bias entries must be finite");
                }
            }
            if !(0.0..=1.0).contains(&self.config.repeat_rate) {
                bail!("repeat rate must be in [0, 1]");
            }
            return Ok(());
        }
        let targets = self.targets()?;
        if targets.len() < 3 {
            bail!("next-token snippet must contain at least two characters");
        }
        if self.config.learning_passes == 0 {
            bail!("next-token learning passes must be positive");
        }
        let mut alphabet = [false; Symbol::COUNT];
        for target in targets {
            alphabet[target.index()] = true;
        }
        if !(Symbol::A.index()..=Symbol::Z.index()).all(|index| alphabet[index]) {
            bail!("next-token snippet must contain every letter a through z");
        }
        if !alphabet[Symbol::Space.index()] {
            bail!("next-token snippet must contain at least one space");
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
        (self.snippet_char_count() + 1) * self.config.learning_passes
    }

    fn start(&self, panel_seed: u64, instance: usize) -> Self::State {
        NextTokenPredictionState {
            targets: if self.config.generalize {
                self.generated_targets(panel_seed, instance)
            } else {
                self.targets().expect("validated next-token snippet")
            },
            position: 0,
            learning_pass: 0,
            pass_exact: true,
            probe_position: 0,
            probe_exact: true,
        }
    }

    fn observe(&self, state: &Self::State) -> Observation {
        Observation {
            symbol: Some(if state.position == 0 {
                Symbol::End
            } else {
                state.targets[state.position - 1]
            }),
        }
    }

    fn step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.targets[state.position];
        let correct = action == expected;
        state.pass_exact &= correct;
        state.position += 1;
        let pass_done = state.position == state.targets.len();
        let trial_outcome = pass_done.then_some(state.pass_exact);
        if pass_done {
            state.learning_pass += 1;
            state.position = 0;
            state.pass_exact = true;
        }
        Transition {
            reward: if correct { 1.0 } else { -1.0 / 27.0 },
            expected_action: Some(expected),
            teaching_target: Some(expected),
            // In predictive-coding mode every correct in-stream prediction is a
            // reproductive ticket (continuous minimal-criterion fitness); the
            // supervised mode still scores only the frozen probe.
            success_events: if self.config.predictive_coding {
                u32::from(correct)
            } else {
                0
            },
            correct,
            trial_outcome,
            done: state.learning_pass == self.config.learning_passes,
        }
    }

    fn probe_steps_per_instance(&self) -> usize {
        self.snippet_char_count() + 1
    }

    fn begin_probe(&self, state: &mut Self::State) {
        state.probe_position = 0;
        state.probe_exact = true;
    }

    fn probe_observe(&self, state: &Self::State) -> Observation {
        Observation {
            symbol: Some(if state.probe_position == 0 {
                Symbol::End
            } else {
                state.targets[state.probe_position - 1]
            }),
        }
    }

    fn probe_step(&self, state: &mut Self::State, action: Symbol) -> Transition {
        let expected = state.targets[state.probe_position];
        let correct = action == expected;
        state.probe_exact &= correct;
        state.probe_position += 1;
        let done = state.probe_position == state.targets.len();
        Transition {
            reward: 0.0,
            expected_action: Some(expected),
            teaching_target: None,
            success_events: u32::from(correct),
            correct,
            trial_outcome: done.then_some(state.probe_exact),
            done,
        }
    }
}
