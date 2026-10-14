//! Decontamination against the evaluation prompts.
//!
//! # The failure this prevents
//!
//! If an evaluation set's text is in the training data, the score measures
//! memorisation instead of language, and **nothing in the pipeline reports it.**
//! The loss curve looks better, the generations look better, and the model has
//! simply been shown the answers.
//!
//! `29_DATA_CLEANING_RESEARCH.md` calls it the field's number one silent failure.
//! It is silent because a contaminated model is not broken — it is better at the
//! test and worse at the task, and only the test is being looked at.
//!
//! # Why the check is a substring match and not an n-gram overlap
//!
//! The usual rule is a thirteen-token overlap, which is long enough that a
//! coincidence is implausible. The evaluation prompts in
//! `configs/eval-prompts-bilingual.md` are two to eight words — `روزی روزگاری`,
//! `Once upon a time` — so **a thirteen-gram never forms from one of them** and
//! the usual rule would report zero contamination on a corpus that contained
//! every prompt verbatim.
//!
//! So the whole prompt is the needle. That is stricter than an n-gram rule, not
//! looser: it matches the prompt exactly rather than a window of it.
//!
//! # What this cannot do
//!
//! It matches the text as written. A corpus document that carries the same words
//! with different normalization — an Arabic yeh where the prompt has a Persian
//! one, a different space — will not match, and the check will report clean.
//!
//! **So the count is a lower bound, and the honest report says so.** Closing that
//! gap means normalizing both sides identically before comparing, which is what
//! [`crate::normalize`] is for and is not wired up here.

use std::collections::HashSet;
use std::path::Path;

use crate::error::DataError;

/// Prompts that training text must not contain, and where each came from.
#[derive(Debug, Clone, Default)]
pub struct Decontamination {
    /// The needles, with a label naming the prompt.
    needles: Vec<(String, String)>,
    /// The same needles normalized, for the comparison.
    normalized: Vec<(String, String)>,
}

impl Decontamination {
    /// An empty set, which matches nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one needle.
    pub fn add(&mut self, label: impl Into<String>, text: impl Into<String>) {
        let label = label.into();
        let text = text.into();
        self.normalized
            .push((label.clone(), normalize_for_matching(&text)));
        self.needles.push((label, text));
    }

    /// Whether a document is contaminated, and by which prompt.
    ///
    /// Returns the label of the first prompt found. A document that contains one
    /// prompt is removed; one that contains several is removed once, and the
    /// caller's counter is what says how many documents were affected rather
    /// than how many matches there were.
    #[must_use]
    pub fn check(&self, text: &str) -> Option<&str> {
        if self.normalized.is_empty() {
            return None;
        }
        let haystack = normalize_for_matching(text);
        self.normalized
            .iter()
            .find(|(_, needle)| haystack.contains(needle.as_str()))
            .map(|(label, _)| label.as_str())
    }

    /// How many needles are loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.needles.len()
    }

    /// Whether nothing is loaded.
    ///
    /// Worth calling before a run. **An empty set matches nothing and reports a
    /// clean corpus**, which is indistinguishable from a corpus that is clean.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.needles.is_empty()
    }

    /// The labels, in the order they were added.
    #[must_use]
    pub fn labels(&self) -> Vec<&str> {
        self.needles
            .iter()
            .map(|(label, _)| label.as_str())
            .collect()
    }

    /// Reads the prompts out of the evaluation document.
    ///
    /// The prompts live in the markdown tables as backticked cells, which makes
    /// the document the single source: a prompt added to the table is checked
    /// against the corpus without anyone remembering to add it in a second
    /// place. Two sources would drift and the drift would be invisible.
    pub fn from_markdown(path: impl AsRef<Path>) -> Result<Self, DataError> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).map_err(|source| DataError::LanguageAdapterRead {
                path: path.to_path_buf(),
                source,
            })?;
        Self::parse_markdown(&text, path)
    }

    /// Parses prompts from markdown text.
    pub fn parse_markdown(text: &str, origin: &Path) -> Result<Self, DataError> {
        let mut set = Self::new();

        for line in text.lines() {
            let trimmed = line.trim();
            // Table rows only. The prose around them contains backticked words
            // too — `روزی روزگاری` is mentioned in the explanation above the
            // table — and taking those as needles would have the corpus checked
            // against its own documentation.
            if !trimmed.starts_with('|') {
                continue;
            }

            let cells: Vec<&str> = trimmed.split('|').map(str::trim).collect();
            // The first and last cells are the empty strings outside the row's
            // outer pipes.
            let Some(&label) = cells.get(1) else { continue };
            let Some(&cell) = cells.get(2) else { continue };

            // The label column is `P1`, `E3` and so on. A header or a separator
            // row does not look like that and is skipped.
            if !is_prompt_label(label) {
                continue;
            }

            // A prompt has two jobs and they pull in opposite directions. As a
            // generation prompt it should be the most natural opening there is;
            // as a contamination needle it has to be rare enough that finding it
            // means something. `Once upon a time` is in 3.8% of a corpus of
            // English books, so it is a good generation prompt and a useless
            // needle, and the table says which each row is.
            //
            // A row without the column is treated as a needle. Silence is the
            // conservative reading: a prompt nobody has classified still gets
            // checked, and the worst outcome is a contaminated document being
            // reported rather than missed.
            // The last cell is the empty string outside the row's closing pipe,
            // and the marker is written in bold, so both are stripped before the
            // comparison rather than being relied on not to be there.
            let marker = cells.iter().rev().find(|cell| !cell.is_empty()).copied();
            if marker
                .is_some_and(|marker| marker.trim_matches('*').trim().eq_ignore_ascii_case("no"))
            {
                continue;
            }

            let Some(prompt) = first_backticked(cell) else {
                continue;
            };
            set.add(label, prompt);
        }

        if set.is_empty() {
            return Err(DataError::LanguageAdapterInvalid {
                path: origin.to_path_buf(),
                reason: "no prompts found; the tables this reads are missing or were reformatted, \
                         and an empty set reports every corpus clean"
                    .to_owned(),
            });
        }

        Ok(set)
    }
}

/// Whether a table's first cell is a prompt label: a letter followed by digits.
fn is_prompt_label(cell: &str) -> bool {
    let mut characters = cell.chars();
    match characters.next() {
        Some(first) if first.is_ascii_uppercase() => {}
        _ => return false,
    }
    let rest: String = characters.collect();
    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
}

/// The contents of the first backticked span in a cell.
fn first_backticked(cell: &str) -> Option<String> {
    let start = cell.find('`')? + 1;
    let rest = &cell[start..];
    let end = rest.find('`')?;
    let prompt = rest[..end].trim();
    if prompt.is_empty() {
        return None;
    }
    Some(prompt.to_owned())
}

/// Folds the differences that are not differences of wording.
///
/// Lowercasing and collapsing runs of whitespace to one space. **Deliberately
/// minimal.** Every fold added here is a fold the corpus side has not had, and a
/// fold applied to one side and not the other turns a match into a miss — so this
/// does the two that are unambiguous and stops. The Persian-specific folds, yeh
/// and kaf and the digits, belong to the normalization stage where both sides get
/// them.
fn normalize_for_matching(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_space = false;
    for character in text.chars() {
        if character.is_whitespace() {
            if !in_space && !out.is_empty() {
                out.push(' ');
            }
            in_space = true;
            continue;
        }
        in_space = false;
        for lower in character.to_lowercase() {
            out.push(lower);
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

/// How many documents a check removed, and by which prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContaminationReport {
    /// Documents checked.
    pub documents: u64,
    /// Documents that contained a prompt.
    pub contaminated: u64,
    /// Removal count per prompt label.
    pub by_label: Vec<(String, u64)>,
}

impl ContaminationReport {
    /// The share of documents removed.
    #[must_use]
    pub fn share(&self) -> f64 {
        if self.documents == 0 {
            0.0
        } else {
            self.contaminated as f64 / self.documents as f64
        }
    }

    /// A one-line summary for a run's log.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} of {} documents removed ({:.4}%)",
            self.contaminated,
            self.documents,
            self.share() * 100.0
        )
    }
}

/// Counts contamination across a corpus.
#[derive(Debug, Default)]
pub struct ContaminationCounter {
    seen: HashSet<String>,
    report: ContaminationReport,
}

impl ContaminationCounter {
    /// An empty counter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks one document and records the outcome.
    pub fn observe(&mut self, text: &str, decontamination: &Decontamination) {
        self.report.documents += 1;
        if let Some(label) = decontamination.check(text) {
            self.report.contaminated += 1;
            if self.seen.insert(label.to_owned()) {
                self.report.by_label.push((label.to_owned(), 0));
            }
            if let Some(entry) = self
                .report
                .by_label
                .iter_mut()
                .find(|(name, _)| name == label)
            {
                entry.1 += 1;
            }
        }
    }

    /// The report so far.
    #[must_use]
    pub fn report(&self) -> &ContaminationReport {
        &self.report
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

    use super::{ContaminationCounter, Decontamination};

    fn set() -> Decontamination {
        let mut set = Decontamination::new();
        set.add("P1", "روزی روزگاری");
        set.add("E1", "Once upon a time");
        set
    }

    #[test]
    fn a_prompt_inside_a_document_is_found() {
        let set = set();
        assert_eq!(
            set.check("روزی روزگاری در یک دهکده کوچک مردی زندگی می‌کرد."),
            Some("P1")
        );
        assert_eq!(set.check("Once upon a time there was a girl."), Some("E1"));
    }

    #[test]
    fn a_clean_document_is_not_flagged() {
        let set = set();
        assert_eq!(
            set.check("این یک متن کاملا بی‌ربط است که هیچ چیزی از پرسش‌ها ندارد."),
            None
        );
    }

    #[test]
    fn an_empty_set_matches_nothing_and_says_so() {
        // The trap: an empty set reports every corpus clean, which is
        // indistinguishable from a corpus that is clean. `is_empty` exists so a
        // run can refuse to proceed on one.
        let empty = Decontamination::new();
        assert!(empty.is_empty());
        assert_eq!(empty.check("روزی روزگاری در یک دهکده کوچک"), None);
        assert!(!set().is_empty());
    }

    #[test]
    fn case_and_spacing_are_folded_on_both_sides() {
        let set = set();
        assert_eq!(set.check("ONCE   UPON   A   TIME there was"), Some("E1"));
        assert_eq!(set.check("once upon a time"), Some("E1"));
    }

    #[test]
    fn the_check_reads_the_real_evaluation_document() {
        // The real file, not a fixture. A reformat that breaks the tables would
        // otherwise leave the check silently matching nothing.
        let path = format!(
            "{}/../../configs/eval-prompts-bilingual.md",
            env!("CARGO_MANIFEST_DIR")
        );
        let set = Decontamination::from_markdown(&path).expect("the evaluation document parses");

        assert!(!set.is_empty());
        assert_eq!(
            set.len(),
            14,
            "sixteen prompts, less the two the table marks as generation-only"
        );

        let labels = set.labels();
        for expected in ["P2", "P4", "P5", "P8", "E2", "E4", "E8"] {
            assert!(labels.contains(&expected), "`{expected}` must be loaded");
        }
        // The two most common openings in either language, which the table marks
        // `no`. Measured on the real corpora: `Once upon a time` was in 3.8 % of
        // English documents, so finding it says nothing about contamination.
        for excluded in ["P1", "E1"] {
            assert!(
                !labels.contains(&excluded),
                "`{excluded}` is too common to be a needle"
            );
        }
        assert_eq!(set.check("Once upon a time there was a girl."), None);
        assert_eq!(set.check("روزی روزگاری در یک دهکده"), None);

        // The two prompts that carry ZWNJ, which are the ones that would break
        // first if the document were reformatted and a backtick were lost.
        assert_eq!(
            set.check("شهر تهران در سال‌های اخیر چه تغییراتی کرده است"),
            Some("P4")
        );
        assert_eq!(set.check("او گفت: «من نمی‌دانم چه کار کنم"), Some("P5"));
    }

    #[test]
    fn the_prose_around_the_tables_is_not_taken_as_a_needle() {
        // The document mentions `روزی روزگاری` in its own explanation. If the
        // parser took every backticked span it would check the corpus against the
        // documentation's prose as well as its prompts.
        let path = format!(
            "{}/../../configs/eval-prompts-bilingual.md",
            env!("CARGO_MANIFEST_DIR")
        );
        let set = Decontamination::from_markdown(&path).expect("parses");
        assert_eq!(set.len(), 14, "only the marked table rows are needles");
    }

    #[test]
    fn a_document_holding_no_prompt_is_kept() {
        let set = set();
        let ordinary = "کتابی که دیروز خواندم درباره تاریخ ایران بود و بسیار جالب.";
        assert_eq!(set.check(ordinary), None);
    }

    #[test]
    fn the_counter_reports_what_it_removed() {
        let set = set();
        let mut counter = ContaminationCounter::new();
        counter.observe("روزی روزگاری در دهکده", &set);
        counter.observe("متن پاک و بی‌ربط", &set);
        counter.observe("Once upon a time there was", &set);
        counter.observe("روزی روزگاری دوباره", &set);

        let report = counter.report();
        assert_eq!(report.documents, 4);
        assert_eq!(report.contaminated, 3);
        assert!((report.share() - 0.75).abs() < 1e-9);
        assert!(report.summary().contains("3 of 4"));
        assert!(report.by_label.iter().any(|(l, n)| l == "P1" && *n == 2));
        assert!(report.by_label.iter().any(|(l, n)| l == "E1" && *n == 1));
    }
}
