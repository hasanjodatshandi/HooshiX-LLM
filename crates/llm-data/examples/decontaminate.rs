//! Checks a corpus slice against the evaluation prompts.
//!
//! # Why this is an example and not a test
//!
//! The unit tests check the mechanism against the real evaluation document, and
//! that is what belongs in the suite. **Whether the actual corpus contains any of
//! the prompts is a fact about the corpus, not about the code** — it changes when
//! the corpus changes, it needs a multi-gigabyte file that is not in the
//! repository, and a test that asserted it would fail on a fresh clone for
//! reasons that have nothing to do with the change being made.
//!
//! So it is a tool. Run it after preparing a corpus, and read the number.
//!
//! ```text
//! cargo run --release -p llm-data --example decontaminate -- \
//!     --prompts configs/eval-prompts-bilingual.md \
//!     --corpus artifacts/tokenizer-data/persian.txt \
//!     --corpus artifacts/tokenizer-data/english.txt
//! ```
//!
//! The corpus files are one document per line, which is what `convert
//! --plain-text` writes and what the tokenizer trainer reads.

#![deny(unsafe_code)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

use llm_data::{ContaminationCounter, Decontamination};

/// The prompts could not be read.
const EXIT_PROMPTS: u8 = 2;
/// A corpus file could not be read.
const EXIT_CORPUS: u8 = 3;

fn main() -> ExitCode {
    let mut prompts: Option<PathBuf> = None;
    let mut corpora: Vec<PathBuf> = Vec::new();
    let mut limit: Option<u64> = None;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--prompts" => prompts = arguments.next().map(PathBuf::from),
            "--corpus" => {
                if let Some(path) = arguments.next() {
                    corpora.push(PathBuf::from(path));
                }
            }
            "--limit" => limit = arguments.next().and_then(|value| value.parse().ok()),
            other => {
                eprintln!("decontaminate: unrecognised argument `{other}`");
                return ExitCode::from(2);
            }
        }
    }

    let Some(prompts) = prompts else {
        eprintln!("decontaminate: --prompts is required");
        return ExitCode::from(EXIT_PROMPTS);
    };
    if corpora.is_empty() {
        eprintln!("decontaminate: at least one --corpus is required");
        return ExitCode::from(EXIT_CORPUS);
    }

    let set = match Decontamination::from_markdown(&prompts) {
        Ok(set) => set,
        Err(error) => {
            eprintln!("decontaminate: {error}");
            return ExitCode::from(EXIT_PROMPTS);
        }
    };

    // Refuse to report a clean corpus from a set that checks nothing. The two are
    // indistinguishable in the output and only one of them is good news.
    if set.is_empty() {
        eprintln!("decontaminate: the prompt set is empty, so every corpus would report clean");
        return ExitCode::from(EXIT_PROMPTS);
    }

    println!("prompts  {}", set.len());
    println!("         {}", set.labels().join(" "));
    println!();

    let mut checked: u64 = 0;
    let mut contaminated: u64 = 0;
    let mut by_label: Vec<(String, u64)> = Vec::new();

    for path in &corpora {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("decontaminate: {}: {error}", path.display());
                return ExitCode::from(EXIT_CORPUS);
            }
        };

        let mut counter = ContaminationCounter::new();
        for line in BufReader::with_capacity(1 << 20, file).lines() {
            let Ok(line) = line else { break };
            if limit.is_some_and(|limit| counter.report().documents >= limit) {
                break;
            }
            counter.observe(&line, &set);
        }

        let report = counter.report();
        checked += report.documents;
        contaminated += report.contaminated;
        for (label, count) in &report.by_label {
            match by_label.iter_mut().find(|(name, _)| name == label) {
                Some(entry) => entry.1 += count,
                None => by_label.push((label.clone(), *count)),
            }
        }

        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        println!("{name:<24} {}", report.summary());
        for (label, count) in &report.by_label {
            println!("    {label:<6} {count}");
        }
    }

    println!();
    if checked == 0 {
        println!("nothing was checked");
        return ExitCode::SUCCESS;
    }
    let share = contaminated as f64 / checked as f64;
    println!(
        "total  {contaminated} of {checked} documents ({:.4}%)",
        share * 100.0
    );

    if contaminated == 0 {
        println!();
        println!("Nothing matched. Read this as a lower bound rather than a proof:");
        println!("the check compares text as written, so a document carrying the same");
        println!("words with different normalization would not be seen.");
        return ExitCode::SUCCESS;
    }

    println!();
    println!("Something matched, and that needs looking at before training. Either the");
    println!("prompts are less distinctive than intended, or the corpus holds them.");
    ExitCode::SUCCESS
}
