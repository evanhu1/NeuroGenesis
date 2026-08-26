use crate::run_output_path;
use anyhow::{anyhow, bail, Result};
use evolution::{
    run_resource_ecology, ActionSelection, AgentEvaluationConfig, AsexualSearchConfig,
    LearningNormalization, LearningRule, ResourceEcologyConfig, ResourceEcologyTask,
    SelectionScheme, SymbolicEcologyAudit, SymbolicEcologyMetrics, TaskEcology,
};
use serde_json::json;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;
use task_library::{
    next_token_prediction::{NextTokenPredictionConfig, NextTokenPredictionTask},
    symbolic_compute::{SymbolicComputeConfig, SymbolicComputeTask},
    SymbolicTask,
};
use types::{OrganismGenome, SeedGenomeConfig};
use views::atomic_write;

const ALGORITHM: &str = "task_ecology_asexual_v1";
const DEFAULT_SEED_CONFIG: &str = config::CANONICAL_SEED_GENOME_CONFIG_PATH;

struct CommonRequest {
    seed_config_path: String,
    seed: u64,
    search: AsexualSearchConfig,
    ecology: ResourceEcologyConfig,
    agent: AgentEvaluationConfig,
    lesion_internal_plasticity: bool,
    co_evolve: bool,
    forge_population: usize,
    forge_snippets: usize,
    task_args: Vec<String>,
}

pub(crate) fn run_cli(args: &[&str], out_dir: &str, out: &mut impl Write) -> Result<()> {
    let Some((&task_name, tail)) = args.split_first() else {
        return write_help(out);
    };
    if matches!(task_name, "help" | "--help" | "-h") {
        return write_help(out);
    }
    if task_name == "analyze" {
        return analyze(tail, out);
    }
    let evaluate_source = if tail.first() == Some(&"evaluate") {
        Some(
            *tail
                .get(1)
                .ok_or_else(|| anyhow!("ecology {task_name} evaluate needs a frozen genome"))?,
        )
    } else {
        None
    };
    let tail = if evaluate_source.is_some() {
        &tail[2..]
    } else if tail.first() == Some(&"run") {
        &tail[1..]
    } else {
        tail
    };
    let plan = tail.first() == Some(&"plan");
    let tail = if plan { &tail[1..] } else { tail };
    let common = parse_common(task_name, tail)?;
    match task_name {
        "next-token" => {
            let task = NextTokenPredictionTask {
                config: parse_next_token(&common.task_args)?,
            };
            if common.co_evolve && evaluate_source.is_none() {
                return execute_coevolve(task, common, out_dir, plan, out);
            }
            dispatch(task, common, out_dir, plan, evaluate_source, out)
        }
        "symbolic" => {
            let task = SymbolicComputeTask {
                config: parse_symbolic(&common.task_args)?,
            };
            dispatch(task, common, out_dir, plan, evaluate_source, out)
        }
        other => bail!(
            "unknown ecology task `{other}`; expected next-token or symbolic"
        ),
    }
}

fn parse_common(task_name: &str, args: &[&str]) -> Result<CommonRequest> {
    let mut agent = AgentEvaluationConfig::default();
    if task_name == "next-token" || task_name == "symbolic" {
        agent.training_instances = 1;
        agent.development_instances = 1;
        agent.sealed_instances = 1;
        agent.training_rollouts = 1;
        agent.development_rollouts = 1;
        agent.sealed_rollouts = 1;
        agent.learning_rule = LearningRule::CategoricalPredictionError;
        agent.action_selection = ActionSelection::Greedy;
    }
    let mut request = CommonRequest {
        seed_config_path: DEFAULT_SEED_CONFIG.to_owned(),
        seed: 0,
        search: AsexualSearchConfig::default(),
        ecology: ResourceEcologyConfig::default(),
        agent,
        lesion_internal_plasticity: false,
        co_evolve: false,
        forge_population: 24,
        forge_snippets: 2,
        task_args: Vec::new(),
    };
    if task_name == "next-token" || task_name == "symbolic" {
        // Next-token stores its transition table in plastic readout weights;
        // the full founder interface is required (fraction dose-response:
        // 0.25/0.5/0.75/1.0 -> 61/70/89/95% sealed). Overridable via --param.
        request.search.initial_connection_fraction = 1.0;
    }
    let mut index = 0;
    while index < args.len() {
        let flag = args[index];
        let raw = args
            .get(index + 1)
            .copied()
            .ok_or_else(|| anyhow!("{flag} needs a value"))?;
        match flag {
            "--seed" => request.seed = raw.parse()?,
            "--population" => request.search.population_size = raw.parse()?,
            "--generations" => request.search.generations = raw.parse()?,
            "--workers" => request.search.evaluation_workers = raw.parse()?,
            "--training-instances" => request.agent.training_instances = raw.parse()?,
            "--development-instances" => request.agent.development_instances = raw.parse()?,
            "--sealed-instances" => request.agent.sealed_instances = raw.parse()?,
            "--training-rollouts" => request.agent.training_rollouts = raw.parse()?,
            "--development-rollouts" => request.agent.development_rollouts = raw.parse()?,
            "--sealed-rollouts" => request.agent.sealed_rollouts = raw.parse()?,
            "--seed-config" => request.seed_config_path = raw.to_owned(),
            "--exact-elites" => request.ecology.exact_elite_copies = raw.parse()?,
            "--tournament-size" => request.ecology.tournament_size = raw.parse()?,
            "--selection" => {
                request.ecology.selection = match raw {
                    "tournament" => SelectionScheme::Tournament,
                    "truncation" => SelectionScheme::Truncation,
                    "proportional" => SelectionScheme::Proportional,
                    _ => bail!("selection must be tournament, truncation, or proportional"),
                }
            }
            "--truncation-survivors" => request.ecology.truncation_survivors = raw.parse()?,
            "--hidden-feedback" => {
                request.agent.hidden_categorical_feedback = match raw {
                    "reward" => false,
                    "categorical" => true,
                    _ => bail!("hidden-feedback must be reward or categorical"),
                }
            }
            "--feedback-channels" => request.search.feedback_channels = raw.parse()?,
            "--temporal-credit" => {
                request.agent.temporal_credit_leaky = match raw {
                    "eprop" => true,
                    "scalar" => false,
                    _ => bail!("temporal-credit must be eprop or scalar"),
                }
            }
            "--exploration-temperature" => request.agent.exploration_temperature = raw.parse()?,
            "--audit-interval" => request.agent.audit_interval = raw.parse()?,
            "--reset-dynamics-at-trial-boundary" => {
                request.agent.reset_dynamics_at_trial_boundary = raw.parse()?
            }
            "--learning-normalization" => {
                request.agent.learning_normalization = match raw {
                    "none" => LearningNormalization::None,
                    "nlms" => LearningNormalization::Nlms,
                    _ => bail!("learning normalization must be none or nlms"),
                }
            }
            "--learning-rule" => {
                request.agent.learning_rule = match raw {
                    "reward_prediction_error" => LearningRule::RewardPredictionError,
                    "categorical_prediction_error" => LearningRule::CategoricalPredictionError,
                    _ => {
                        bail!("learning rule must be reward_prediction_error or categorical_prediction_error")
                    }
                }
            }
            "--action-selection" => {
                request.agent.action_selection = match raw {
                    "greedy" => ActionSelection::Greedy,
                    "sampled" => ActionSelection::Sampled,
                    _ => bail!("action selection must be greedy or sampled"),
                }
            }
            "--lesion" => match raw {
                "internal-plasticity" => request.lesion_internal_plasticity = true,
                _ => bail!("lesion must be internal-plasticity"),
            },
            "--coevolve" => request.co_evolve = raw.parse()?,
            "--forge-population" => request.forge_population = raw.parse()?,
            "--forge-snippets" => request.forge_snippets = raw.parse()?,
            "--param" => {
                let (key, value) = raw
                    .split_once('=')
                    .ok_or_else(|| anyhow!("expected key=value"))?;
                apply_search_param(&mut request.search, key, value)?;
            }
            _ => {
                request.task_args.push(flag.to_owned());
                request.task_args.push(raw.to_owned());
            }
        }
        index += 2;
    }
    request.search.validate()?;
    request.ecology.validate(request.search.population_size)?;
    Ok(request)
}

fn parse_next_token(args: &[String]) -> Result<NextTokenPredictionConfig> {
    let mut config = NextTokenPredictionConfig::default();
    parse_pairs(args, |flag, raw| {
        match flag {
            "--snippet" => config.snippet = raw.to_owned(),
            "--learning-passes" => config.learning_passes = raw.parse()?,
            "--predictive-coding" => config.predictive_coding = raw.parse()?,
            "--generalize" => config.generalize = raw.parse()?,
            "--snippet-length" => config.snippet_length = raw.parse()?,
            other => bail!("unknown next-token option `{other}`"),
        }
        Ok(())
    })?;
    Ok(config)
}


fn parse_symbolic(args: &[String]) -> Result<SymbolicComputeConfig> {
    let mut config = SymbolicComputeConfig::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ops" => {
                config.ops = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--ops needs a value"))?
                    .split(',')
                    .map(|s| s.to_string())
                    .collect();
                i += 2;
            }
            "--word-len" => {
                config.word_len = args.get(i+1).ok_or_else(|| anyhow!("--word-len needs a value"))?.parse()?;
                i += 2;
            }
            "--demos" => {
                config.n_demos = args.get(i+1).ok_or_else(|| anyhow!("--demos needs a value"))?.parse()?;
                i += 2;
            }
            "--queries" => {
                config.n_probes = args.get(i+1).ok_or_else(|| anyhow!("--queries needs a value"))?.parse()?;
                i += 2;
            }
            "--learning-passes" => {
                config.learning_passes = args.get(i+1).ok_or_else(|| anyhow!("--learning-passes needs a value"))?.parse()?;
                i += 2;
            }
            "--predictive-coding" => {
                config.predictive_coding = args.get(i+1).ok_or_else(|| anyhow!("--predictive-coding needs a value"))?.parse()?;
                i += 2;
            }
            other => bail!("unknown symbolic option `{other}`"),
        }
    }
    Ok(config)
}
fn parse_pairs(args: &[String], mut parse: impl FnMut(&str, &str) -> Result<()>) -> Result<()> {
    if !args.len().is_multiple_of(2) {
        bail!("task options require flag/value pairs");
    }
    for pair in args.chunks_exact(2) {
        parse(&pair[0], &pair[1])?;
    }
    Ok(())
}

fn dispatch<T: SymbolicTask + Clone>(
    task: T,
    request: CommonRequest,
    out_dir: &str,
    plan: bool,
    evaluate_source: Option<&str>,
    out: &mut impl Write,
) -> Result<()> {
    task.validate()?;
    if let Some(source) = evaluate_source {
        return evaluate_frozen(
            TaskEcology::new(task, request.agent),
            source,
            request.seed,
            request.lesion_internal_plasticity,
            out,
        );
    }
    if request.lesion_internal_plasticity {
        bail!("--lesion is only valid with ecology TASK evaluate RESULT");
    }
    if plan {
        return writeln!(out, "{}", json!({
            "mode":"task_ecology_plan", "algorithm":ALGORITHM, "task":task.name(),
            "search":request.search, "ecology":request.ecology, "agent":request.agent,
            "task_config":task.config(),
            "maximum_task_steps": request.search.population_size as u128 * request.search.generations as u128 * request.agent.training_instances as u128 * request.agent.training_rollouts as u128 * (task.max_steps_per_instance() + task.probe_steps_per_instance()) as u128,
        })).map_err(Into::into);
    }
    execute(
        TaskEcology::new(task, request.agent.clone()),
        request,
        out_dir,
        out,
    )
}

fn execute_coevolve(
    task: NextTokenPredictionTask,
    request: CommonRequest,
    out_dir: &str,
    plan: bool,
    out: &mut impl Write,
) -> Result<()> {
    use evolution::run_co_evolution_next_token;
    let seed_genome = load_seed_config(&request.seed_config_path)?;
    let total_generations = request.search.generations;
    let started = Instant::now();
    eprintln!(
        "{}",
        json!({"event":"task_ecology_started","task":task.name(),"mode":"co_evolution_text_forge_v1",
               "population":request.search.population_size,"generations":total_generations,
               "workers":request.search.evaluation_workers,"forge_population":request.forge_population,
               "forge_snippets":request.forge_snippets})
    );
    if plan {
        return writeln!(out, "{}", json!({
            "mode":"task_ecology_plan","algorithm":ALGORITHM,"extension":"co_evolution_text_forge_v1",
            "task":task.name(),"search":request.search,"ecology":request.ecology,"agent":request.agent,
            "task_config":task.config(),
            "forge_population":request.forge_population,"forge_snippets":request.forge_snippets,
        })).map_err(Into::into);
    }
    let (result, trajectory) = run_co_evolution_next_token(
        task.config.clone(),
        request.search,
        request.ecology,
        seed_genome,
        request.seed,
        request.agent.clone(),
        request.forge_population,
        request.forge_snippets,
        |generation| {
            eprintln!(
                "{}",
                json!({
                    "event":"task_ecology_generation","task":task.name(),"generation":generation.generation,
                    "leading_accuracy":generation.leading_evaluation.accuracy,
                    "leading_trial_success_rate":generation.leading_evaluation.trial_success_rate,
                    "development_accuracy":generation.leading_audit.as_ref().map(|audit| audit.primary.accuracy),
                    "hidden_nodes":generation.leading_hidden_nodes,
                    "enabled_connections":generation.leading_enabled_connections,
                })
            );
        },
    )?;
    let mut path = run_output_path(out_dir, &format!("task-ecology-{}", result.task))?;
    path.set_extension("json.zst");
    let path_string = path.to_string_lossy().into_owned();
    atomic_write(&path_string, |writer| {
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        serde_json::to_writer(&mut encoder, &result)?;
        encoder.finish()?;
        Ok(())
    })?;
    let mut forge_path = path.clone();
    let stem = forge_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    forge_path.set_file_name(format!("{stem}.forges.json"));
    atomic_write(&forge_path.to_string_lossy(), |writer| {
        serde_json::to_writer(writer, &trajectory)?;
        Ok(())
    })?;
    writeln!(
        out,
        "{}",
        json!({
            "wrote":path_string,"forges_wrote":forge_path.to_string_lossy(),
            "algorithm":"co_evolution_text_forge_v1","task":result.task,
            "termination":result.termination,
            "selected_generation":result.selected_generation,
            "development":audit_summary(&result.selected_development_audit),
            "sealed":audit_summary(&result.sealed_audit),
            "work":result.work,"wall_time_seconds":result.total_wall_time_seconds,
            "elapsed_seconds":started.elapsed().as_secs_f64(),
        })
    )?;
    Ok(())
}

fn execute<T: SymbolicTask + Clone>(
    task: TaskEcology<T>,
    request: CommonRequest,
    out_dir: &str,
    out: &mut impl Write,
) -> Result<()> {
    let seed_genome = load_seed_config(&request.seed_config_path)?;
    let total_generations = request.search.generations;
    let population = request.search.population_size;
    let started = Instant::now();
    eprintln!(
        "{}",
        json!({"event":"task_ecology_started","task":task.task.name(),"population":population,"generations":total_generations,"workers":request.search.evaluation_workers})
    );
    let result = run_resource_ecology(
        &task,
        request.search,
        request.ecology,
        seed_genome,
        request.seed,
        |generation| {
            let completed = generation.generation + 1;
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "{}",
                json!({
                    "event":"task_ecology_generation","task":task.task.name(),"generation":generation.generation,
                    "completed_generations":completed,"total_generations":total_generations,
                    "progress_percent":100.0*f64::from(completed)/f64::from(total_generations),
                    "leading_accuracy":generation.leading_evaluation.accuracy,
                    "leading_trial_success_rate":generation.leading_evaluation.trial_success_rate,
                    "leading_resource_units":generation.leading_evaluation.resource_units,
                    "development_accuracy":generation.leading_audit.as_ref().map(|audit| audit.primary.accuracy),
                    "hidden_nodes":generation.leading_hidden_nodes,"enabled_connections":generation.leading_enabled_connections,
                    "elapsed_seconds":elapsed,"eta_seconds":elapsed/f64::from(completed)*f64::from(total_generations-completed),
                })
            );
        },
    )?;
    let mut path = run_output_path(out_dir, &format!("task-ecology-{}", task.task.name()))?;
    path.set_extension("json.zst");
    let path_string = path.to_string_lossy().into_owned();
    atomic_write(&path_string, |writer| {
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        serde_json::to_writer(&mut encoder, &result)?;
        encoder.finish()?;
        Ok(())
    })?;
    writeln!(
        out,
        "{}",
        json!({
            "wrote":path_string,"algorithm":result.algorithm,"task":result.task,"termination":result.termination,
            "selected_generation":result.selected_generation,"development":audit_summary(&result.selected_development_audit),
            "sealed":audit_summary(&result.sealed_audit),"work":result.work,"wall_time_seconds":result.total_wall_time_seconds,
        })
    )?;
    Ok(())
}

fn evaluate_frozen<T: SymbolicTask + Clone>(
    task: TaskEcology<T>,
    source: &str,
    seed: u64,
    lesion_internal_plasticity: bool,
    out: &mut impl Write,
) -> Result<()> {
    task.validate()?;
    let value: serde_json::Value = serde_json::from_reader(result_reader(source)?)?;
    let genome_value = value
        .get("selected_genome")
        .or_else(|| value.get("genome"))
        .unwrap_or(&value)
        .clone();
    let mut genome: OrganismGenome = serde_json::from_value(genome_value)
        .map_err(|error| anyhow!("cannot decode genome from `{source}`: {error}"))?;
    if lesion_internal_plasticity {
        for node in &mut genome.brain.hidden_nodes {
            node.plasticity_receptor = 0.0;
        }
    }
    let audit = task.audit(&genome, "sealed", seed)?;
    writeln!(
        out,
        "{}",
        json!({
            "mode":"task_ecology_frozen_evaluation",
            "source":source,
            "task":task.task.name(),
            "task_config":task.task.config(),
            "agent":task.agent,
            "hidden_nodes":genome.hidden_node_count(),
            "enabled_connections":genome.enabled_connection_count(),
            "lesion":lesion_internal_plasticity.then_some("internal-plasticity"),
            "sealed":audit_summary(&audit),
        })
    )?;
    Ok(())
}

fn audit_summary(audit: &SymbolicEcologyAudit) -> serde_json::Value {
    json!({"cohort":audit.cohort,"primary":metrics(&audit.primary)})
}
fn metrics(value: &SymbolicEcologyMetrics) -> serde_json::Value {
    json!({"accuracy":value.accuracy,"learning_accuracy":value.learning_accuracy,"probe_accuracy":value.probe_accuracy,"exact_string_rate":value.trial_success_rate,"successful_trials":value.successful_trials,"completed_trials":value.completed_trials,"mean_probe_target_probability":value.mean_probe_target_probability,"mean_probe_sequence_probability":value.mean_probe_sequence_probability,"resource_units":value.resource_units,"resource_throughput_per_1000_ticks":value.resource_throughput_per_1000_ticks,"mean_reward":value.mean_reward})
}

fn analyze(args: &[&str], out: &mut impl Write) -> Result<()> {
    if args.is_empty() {
        bail!("ecology analyze needs at least one result");
    }
    let values = args.iter().map(|path| -> Result<_> { let value: serde_json::Value = serde_json::from_reader(result_reader(path)?)?; Ok(json!({"path":path,"task":value["task"],"algorithm":value["algorithm"],"termination":value["termination"],"selected_generation":value["selected_generation"],"sealed_audit":value["sealed_audit"],"work":value["work"]})) }).collect::<Result<Vec<_>>>()?;
    writeln!(
        out,
        "{}",
        if values.len() == 1 {
            values.into_iter().next().unwrap()
        } else {
            json!({"runs":values})
        }
    )?;
    Ok(())
}

fn load_seed_config(path: &str) -> Result<SeedGenomeConfig> {
    config::load_seed_genome_config_from_path(Path::new(path))
}
fn result_reader(path: &str) -> Result<Box<dyn Read>> {
    let file = File::open(path)?;
    if path.ends_with(".zst") {
        Ok(Box::new(zstd::stream::read::Decoder::new(file)?))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn apply_search_param(config: &mut AsexualSearchConfig, key: &str, value: &str) -> Result<()> {
    match key {
        "mutate_weight_probability" => config.mutate_weight_probability = value.parse()?,
        "replace_weight_probability" => config.replace_weight_probability = value.parse()?,
        "weight_perturb_stddev" => config.weight_perturb_stddev = value.parse()?,
        "mutate_bias_probability" => config.mutate_bias_probability = value.parse()?,
        "bias_perturb_stddev" => config.bias_perturb_stddev = value.parse()?,
        "mutate_time_constant_probability" => {
            config.mutate_time_constant_probability = value.parse()?
        }
        "time_constant_perturb_stddev" => config.time_constant_perturb_stddev = value.parse()?,
        "mutate_learning_rate_probability" => {
            config.mutate_learning_rate_probability = value.parse()?
        }
        "learning_rate_perturb_stddev" => config.learning_rate_perturb_stddev = value.parse()?,
        "mutate_plasticity_coefficient_probability" => {
            config.mutate_plasticity_coefficient_probability = value.parse()?
        }
        "plasticity_coefficient_perturb_stddev" => {
            config.plasticity_coefficient_perturb_stddev = value.parse()?
        }
        "mutate_plasticity_receptor_probability" => {
            config.mutate_plasticity_receptor_probability = value.parse()?
        }
        "plasticity_receptor_perturb_stddev" => {
            config.plasticity_receptor_perturb_stddev = value.parse()?
        }
        "add_connection_probability" => config.add_connection_probability = value.parse()?,
        "delete_connection_probability" => config.delete_connection_probability = value.parse()?,
        "add_node_probability" => config.add_node_probability = value.parse()?,
        "delete_node_probability" => config.delete_node_probability = value.parse()?,
        "mutate_only_active_interface" => config.mutate_only_active_interface = value.parse()?,
        "recurrent_node_self_connection" => {
            config.recurrent_node_self_connection = value.parse()?
        }
        "self_recurrent_hidden" => config.self_recurrent_hidden = value.parse()?,
        "dense_recurrence" => config.dense_recurrence = value.parse()?,
        "heterogeneous_time_constants" => config.heterogeneous_time_constants = value.parse()?,
        "mutate_activation_probability" => config.mutate_activation_probability = value.parse()?,
        "initial_connection_fraction" => config.initial_connection_fraction = value.parse()?,
        "input_delay_line_depth" => config.input_delay_line_depth = value.parse()?,
        "add_delay_relay_probability" => config.add_delay_relay_probability = value.parse()?,
        other => bail!("unknown search parameter `{other}`"),
    }
    Ok(())
}

fn write_help(out: &mut impl Write) -> Result<()> {
    writeln!(out, "Task ecology:\n  cli ecology next-token [run|plan] [OPTIONS]\n  cli ecology symbolic [run|plan] [OPTIONS]\n  cli ecology TASK evaluate RESULT [OPTIONS] [--lesion internal-plasticity]\n  cli ecology analyze RESULT...\n\nShared: --seed N --population N --generations N --workers N --training-instances N --development-instances N --sealed-instances N --training-rollouts N --development-rollouts N --sealed-rollouts N --exact-elites N --tournament-size N --selection tournament|truncation|proportional --truncation-survivors N --exploration-temperature F --action-selection greedy|sampled --learning-rule reward_prediction_error|categorical_prediction_error --temporal-credit eprop|scalar --hidden-feedback reward|categorical --learning-normalization none|nlms --reset-dynamics-at-trial-boundary true|false --audit-interval N --param key=value\nSearch --param keys include self_recurrent_hidden, dense_recurrence, heterogeneous_time_constants (all default false), mutate_activation_probability (default 0.3), initial_connection_fraction (founder interface fraction, default 0.25; next-token presets 1.0), add_delay_relay_probability (temporal-memory operator, default 0.05), and input_delay_line_depth (seeded delay line, default 0 = discover it instead).\nNext token: --snippet TEXT --learning-passes N --predictive-coding true|false --generalize true|false --snippet-length N\nSymbolic: --ops copy,reve,rota,dupl,cyph --word-len N --demos N --queries N --learning-passes N").map_err(Into::into)
}
