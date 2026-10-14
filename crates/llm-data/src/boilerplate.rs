//! Boilerplate removal, keyed on the source host.
//!
//! # The case this exists for
//!
//! The first document of the real FineWeb2 Persian shard opens like this:
//!
//! ```text
//! +
//! نوشته شده در شنبه بیستم خرداد 1391ساعت 15:13 توسط behrouz
//!
//!
//! درباره
//! با اسپیکر روشن وارد شوید!!! ...
//! ```
//!
//! A stray `+`, a byline, a navigation word, and then the article. That byline is
//! **five words long**, and it is on every page of that site.
//!
//! # Why frequency, and not near-duplicate detection
//!
//! Five words never form a nine-token shingle, so MinHash is blind to it. What
//! makes it removable is not its shape but its **repetition**: a line that
//! appears on a fifth of a site's pages is furniture, whatever it says.
//!
//! # Why per host
//!
//! A site's furniture is a property of the site. Counting across the whole corpus
//! would need the same byline to appear on many sites before it registered, which
//! for a byline it never does. Counting within a host catches it on the second
//! page. A corpus-wide count is kept as well for the genuinely universal lines —
//! a cookie notice, a licence footer — because those appear once per site and
//! still amount to a great many documents.
//!
//! # The cost of being aggressive
//!
//! A threshold that removes site furniture will also remove a line that the site
//! legitimately repeats — a quoted passage, a standard disclaimer, the name of a
//! recurring column. There is no way to tell those apart by frequency alone.
//!
//! **So the detector reports the lines it removed most, with their text.** A
//! reader can then see whether it took furniture or content, which is the only
//! honest way to run an aggressive filter: **aggression with a visible ledger.**

use std::collections::HashMap;

/// Share of a host's documents a line must appear in to be its furniture.
///
/// A fifth. Safe to set low because the count is per host, where genuine
/// repetition is rare — a site that repeats a sentence on a fifth of its pages is
/// almost always repeating furniture.
pub const PER_HOST_SHARE: f64 = 0.20;

/// Share of the whole corpus a line must appear in to be universal furniture.
///
/// A twentieth, which is a high bar: a line on five percent of a web corpus is on
/// millions of pages. This catches the notices that appear once per site.
pub const CORPUS_SHARE: f64 = 0.05;

/// A corpus needs at least this many documents before a share means anything.
///
/// The corpus rule is a share — five percent — and a share of a small corpus is
/// not a share of anything. In a twenty-document run the floor computes to
/// `ceil(20 * 0.05) = 1`, so **every line is universal boilerplate** and the
/// whole corpus is deleted. This floor is what stops that, and it is the same
/// reasoning as `MIN_DOCUMENTS_PER_HOST`: a statistic needs a population before
/// it means anything, and the failure when it has none is silent and total.
pub const MIN_DOCUMENTS_FOR_CORPUS_RULE: u32 = 100;

/// A host needs at least this many documents before its own statistics mean
/// anything.
///
/// Without a floor, a host with three pages has a line on two of them at sixty
/// seven percent and loses it. The floor is what keeps the rule from eating small
/// sites.
pub const MIN_DOCUMENTS_PER_HOST: u32 = 20;

/// Keep the text of lines seen in at least this many documents.
///
/// Only for the report. Holding the text of every line in a corpus would cost
/// more than the corpus, and a line seen once is not going to be removed, so its
/// text is not worth keeping.
const TEXT_SAMPLE_THRESHOLD: u32 = 5;

/// How many removed lines the report names.
pub const REPORTED_LINES: usize = 20;

/// One line the detector decided was furniture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovedLine {
    /// The line's text, if it was seen often enough to have been kept.
    pub text: String,
    /// Documents it appeared in, corpus-wide.
    pub documents: u32,
    /// Whether it was removed for being common on its own host or everywhere.
    pub scope: RemovalScope,
}

/// Which threshold a line crossed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovalScope {
    /// Common on the host it came from.
    PerHost,
    /// Common across the whole corpus.
    Corpus,
}

/// What a completed boilerplate pass found.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoilerplateReport {
    /// Documents observed.
    pub documents: u32,
    /// Distinct hosts observed.
    pub hosts: usize,
    /// Distinct lines tracked.
    pub distinct_lines: usize,
    /// Lines removed by the host rule and not the corpus rule.
    pub per_host_lines: usize,
    /// Lines removed by the corpus rule.
    pub corpus_lines: usize,
    /// The lines removed from the most documents, for a reader to check.
    ///
    /// **This is the part that makes an aggressive filter defensible.** The
    /// threshold removes repetition without knowing whether the repetition was
    /// furniture or content, and no threshold can. Naming the worst offenders
    /// lets a reader see which it was.
    pub worst: Vec<RemovedLine>,
}

/// Counts line repetition and then removes what repeats.
///
/// Two phases, because a frequency cannot be known until the corpus has been
/// read. [`observe`](Self::observe) accumulates; [`finish`](Self::finish)
/// decides; [`clean`](Self::clean) applies. Reading the corpus twice is the price
/// and it is the only correct order.
#[derive(Debug, Default)]
pub struct Boilerplate {
    /// Line hash to the number of documents it appeared in, corpus-wide.
    corpus: HashMap<u64, u32>,
    /// Host to line hash to document count.
    hosts: HashMap<String, HashMap<u64, u32>>,
    /// Host to documents observed.
    host_documents: HashMap<String, u32>,
    /// Line hash to its text, for lines worth reporting.
    texts: HashMap<u64, String>,
    /// Documents observed.
    documents: u32,
    /// Lines decided as boilerplate, by hash.
    removed: HashMap<u64, RemovalScope>,
}

impl Boilerplate {
    /// An empty detector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one document and the host it came from.
    ///
    /// `host` is `None` when the source has no URL, which is the case for a book
    /// or a code file. Such a document is counted corpus-wide and is not counted
    /// against any host, because attributing it to one would invent a source.
    pub fn observe(&mut self, host: Option<&str>, text: &str) {
        self.documents += 1;
        if let Some(host) = host {
            *self.host_documents.entry(host.to_owned()).or_insert(0) += 1;
        }

        // A line is counted once per document however many times it appears in
        // it. A page that repeats its own footer three times is one document
        // with a footer, not three.
        let mut seen_here: Vec<u64> = Vec::new();
        for line in lines(text) {
            let hash = hash_line(line);
            if seen_here.contains(&hash) {
                continue;
            }
            seen_here.push(hash);

            let count = self.corpus.entry(hash).or_insert(0);
            *count += 1;
            if *count == TEXT_SAMPLE_THRESHOLD {
                self.texts.insert(hash, line.to_owned());
            }

            if let Some(host) = host {
                *self
                    .hosts
                    .entry(host.to_owned())
                    .or_default()
                    .entry(hash)
                    .or_insert(0) += 1;
            }
        }
    }

    /// Decides which lines are furniture.
    ///
    /// Returns the report, and the detector keeps the decision for
    /// [`clean`](Self::clean).
    pub fn finish(mut self) -> (Self, BoilerplateReport) {
        let mut report = BoilerplateReport {
            documents: self.documents,
            hosts: self.host_documents.len(),
            distinct_lines: self.corpus.len(),
            ..BoilerplateReport::default()
        };

        // The corpus-wide rule first, so a line that is both is reported as
        // universal rather than as one site's furniture.
        if self.documents >= MIN_DOCUMENTS_FOR_CORPUS_RULE {
            let floor = (f64::from(self.documents) * CORPUS_SHARE).ceil() as u32;
            for (&hash, &count) in &self.corpus {
                if count >= floor {
                    self.removed.insert(hash, RemovalScope::Corpus);
                }
            }
        }
        report.corpus_lines = self.removed.len();

        for (host, lines) in &self.hosts {
            let documents = self.host_documents.get(host).copied().unwrap_or(0);
            if documents < MIN_DOCUMENTS_PER_HOST {
                continue;
            }
            let floor = (f64::from(documents) * PER_HOST_SHARE).ceil() as u32;
            for (&hash, &count) in lines {
                if count >= floor && !self.removed.contains_key(&hash) {
                    self.removed.insert(hash, RemovalScope::PerHost);
                }
            }
        }
        report.per_host_lines = self.removed.len() - report.corpus_lines;

        // The ledger. Sorted by how many documents the line was taken out of, so
        // the report leads with whatever the filter was most aggressive about.
        let mut worst: Vec<RemovedLine> = self
            .removed
            .iter()
            .filter_map(|(&hash, &scope)| {
                self.texts.get(&hash).map(|text| RemovedLine {
                    text: text.clone(),
                    documents: self.corpus.get(&hash).copied().unwrap_or(0),
                    scope,
                })
            })
            .collect();
        worst.sort_by(|a, b| b.documents.cmp(&a.documents).then(a.text.cmp(&b.text)));
        worst.truncate(REPORTED_LINES);
        report.worst = worst;

        (self, report)
    }

    /// Whether a line is furniture.
    #[must_use]
    pub fn is_removed(&self, line: &str) -> bool {
        self.removed.contains_key(&hash_line(line))
    }

    /// How many documents a line appeared in, corpus-wide.
    ///
    /// Exposed because it is the number the decision was made from, and a report
    /// that names a removed line without it cannot be checked.
    #[must_use]
    pub fn corpus_line_documents(&self, line: &str) -> u32 {
        self.corpus.get(&hash_line(line)).copied().unwrap_or(0)
    }

    /// How many distinct hosts were counted.
    #[must_use]
    pub fn hosts_tracked(&self) -> usize {
        self.hosts.len()
    }

    /// Removes furniture from a document.
    ///
    /// A line is removed whole, and a document that becomes empty is left empty
    /// rather than dropped here — the filter counts what it drops and why, and a
    /// cleaner that quietly removed documents would make those counts wrong.
    #[must_use]
    pub fn clean(&self, text: &str) -> String {
        let mut kept: Vec<&str> = Vec::new();
        for line in text.lines() {
            if self.is_removed(line) {
                continue;
            }
            kept.push(line);
        }
        kept.join("\n")
    }
}

/// The lines of a document that are candidates for removal.
///
/// Blank lines are skipped: they are not furniture, they are separators, and
/// counting them would make every document share one.
fn lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines().map(str::trim).filter(|line| !line.is_empty())
}

/// Hashes a line for counting.
///
/// FNV-1a, written out rather than pulled in. The hash only has to be stable
/// within one run and cheap — a collision merges two lines that would then be
/// removed together, which is a small and acceptable error for a frequency
/// count, and `std`'s default hasher is not guaranteed stable across runs, which
/// would make a two-pass removal disagree with itself.
fn hash_line(line: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in line.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Extracts the host from a URL, for keying the per-host rule.
///
/// Deliberately not a URL parser. It reads the authority section and stops at the
/// first `/`, `?` or `#`, lowercases it, and strips a leading `www.` so that
/// `www.example.com` and `example.com` are the same site. Anything more is a
/// dependency this does not need, and a document whose URL will not parse simply
/// gets no host and is counted corpus-wide.
#[must_use]
pub fn host_of(url: &str) -> Option<String> {
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();

    // Strip userinfo and port. A port is not part of a site's identity for this
    // purpose, and userinfo would otherwise become the host.
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = host.split(':').next().unwrap_or(host);
    let host = host.strip_prefix("www.").unwrap_or(host);

    if host.is_empty() || !host.contains('.') {
        return None;
    }
    Some(host.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

    use super::{
        Boilerplate, MIN_DOCUMENTS_FOR_CORPUS_RULE, MIN_DOCUMENTS_PER_HOST, RemovalScope, host_of,
    };

    /// The byline from the real Persian shard, which is five words and therefore
    /// invisible to a nine-token shingle.
    const BYLINE: &str = "نوشته شده در شنبه بیستم خرداد 1391ساعت 15:13 توسط behrouz";

    fn article(n: usize) -> String {
        format!(
            "{BYLINE}\nدرباره\nمتن شماره {n} این یک مقاله معمولی است که چند جمله دارد \
             و هر جمله چیز متفاوتی می‌گوید. هیچ کدام از این جمله‌ها تکراری نیستند."
        )
    }

    #[test]
    fn a_url_yields_its_host() {
        assert_eq!(
            host_of("http://dooshvari.blogfa.com/post/1").as_deref(),
            Some("dooshvari.blogfa.com")
        );
        assert_eq!(
            host_of("https://www.example.com/a?b=1").as_deref(),
            Some("example.com")
        );
        assert_eq!(
            host_of("https://example.com:8443/x").as_deref(),
            Some("example.com")
        );
        assert_eq!(
            host_of("https://user@example.com/x").as_deref(),
            Some("example.com")
        );
        assert_eq!(host_of("EXAMPLE.COM/x").as_deref(), Some("example.com"));
    }

    #[test]
    fn a_url_without_a_host_yields_none_rather_than_a_guess() {
        // Better no host than a wrong one: a wrong host puts a document under
        // another site's statistics and removes that site's lines from it.
        assert_eq!(host_of(""), None);
        assert_eq!(host_of("localhost/x"), None);
        assert_eq!(host_of("not a url at all"), None);
    }

    #[test]
    fn a_byline_repeated_across_a_host_is_removed() {
        let mut detector = Boilerplate::new();
        for n in 0..MIN_DOCUMENTS_PER_HOST {
            detector.observe(Some("blogfa.com"), &article(n as usize));
        }
        let (detector, report) = detector.finish();

        assert!(
            report.per_host_lines >= 2,
            "the byline and درباره are both furniture"
        );
        assert!(detector.is_removed(BYLINE));
        assert!(detector.is_removed("درباره"));

        let cleaned = detector.clean(&article(999));
        assert!(!cleaned.contains("behrouz"), "the byline is gone");
        assert!(!cleaned.contains("درباره"), "the navigation word is gone");
        assert!(cleaned.contains("مقاله معمولی"), "the article survives");
    }

    #[test]
    fn a_document_with_no_url_is_counted_but_attributed_to_no_host() {
        let mut detector = Boilerplate::new();
        for n in 0..MIN_DOCUMENTS_PER_HOST {
            detector.observe(None, &article(n as usize));
        }
        let (_, report) = detector.finish();
        assert_eq!(report.documents, MIN_DOCUMENTS_PER_HOST);
        assert_eq!(report.hosts, 0);
        // With no host there is no per-host statistic, and the corpus rule needs
        // five percent of the corpus, which these documents do not reach. So
        // nothing is removed, which is the honest answer for an unkeyed source.
        assert_eq!(report.per_host_lines, 0);
    }

    #[test]
    fn a_small_corpus_has_no_universal_lines() {
        // A share of a small corpus is not a share of anything. Twenty documents
        // put the corpus floor at one, and every line would be furniture — the
        // silent, total failure the floor exists to prevent.
        let mut detector = Boilerplate::new();
        for n in 0..MIN_DOCUMENTS_FOR_CORPUS_RULE - 1 {
            detector.observe(Some(&format!("h{n}.example")), &article(n as usize));
        }
        let (_, report) = detector.finish();
        assert_eq!(
            report.corpus_lines, 0,
            "a corpus below the floor has no universal lines"
        );
    }

    #[test]
    fn a_small_host_is_left_alone() {
        // Three pages is not evidence. Without the floor, a line on two of them
        // is at sixty-seven percent and would be removed from a site that has
        // barely any pages to judge it by.
        let mut detector = Boilerplate::new();
        for n in 0..3 {
            detector.observe(Some("tiny.example"), &article(n));
        }
        let (detector, report) = detector.finish();
        assert_eq!(report.per_host_lines, 0);
        assert!(!detector.is_removed(BYLINE));
    }

    #[test]
    fn one_hosts_furniture_is_not_removed_from_another() {
        // The whole reason the count is per host. `blogfa.com` repeats its byline
        // on every page; `other.example` says something different on each, and
        // must not lose a line because a different site repeats it.
        let mut detector = Boilerplate::new();
        for n in 0..MIN_DOCUMENTS_PER_HOST {
            detector.observe(Some("blogfa.com"), &article(n as usize));
        }
        for n in 0..MIN_DOCUMENTS_PER_HOST {
            detector.observe(
                Some("other.example"),
                &format!("متن یکتای شماره {n} که در هیچ جای دیگری نیست و تکرار نمی‌شود."),
            );
        }
        let (detector, _) = detector.finish();

        let unique = "متن یکتای شماره 7 که در هیچ جای دیگری نیست و تکرار نمی‌شود.";
        assert!(
            !detector.is_removed(unique),
            "a unique line is never furniture"
        );
        assert_eq!(detector.clean(unique), unique);
    }

    #[test]
    fn a_line_within_one_document_is_counted_once() {
        // A page that prints its own footer three times is one document with a
        // footer, not three documents' worth. Counting occurrences rather than
        // documents would let a single page make its own lines furniture.
        let mut detector = Boilerplate::new();
        detector.observe(
            Some("x.example"),
            &format!("{BYLINE}\n{BYLINE}\n{BYLINE}\nمتن"),
        );
        let (detector, _) = detector.finish();
        assert_eq!(detector.corpus_line_documents(BYLINE), 1);
    }

    #[test]
    fn the_report_names_what_was_removed_most() {
        // The ledger that makes an aggressive filter defensible. A reader has to
        // be able to see whether the filter took furniture or a quoted passage.
        let mut detector = Boilerplate::new();
        for n in 0..MIN_DOCUMENTS_PER_HOST {
            detector.observe(Some("blogfa.com"), &article(n as usize));
        }
        let (_, report) = detector.finish();

        assert!(
            !report.worst.is_empty(),
            "the report must name what it removed"
        );
        let byline = report
            .worst
            .iter()
            .find(|line| line.text == BYLINE)
            .expect("the byline must be named in the report");
        assert_eq!(byline.documents, MIN_DOCUMENTS_PER_HOST);
        assert_eq!(byline.scope, RemovalScope::PerHost);
    }

    #[test]
    fn cleaning_is_idempotent() {
        let mut detector = Boilerplate::new();
        for n in 0..MIN_DOCUMENTS_PER_HOST {
            detector.observe(Some("blogfa.com"), &article(n as usize));
        }
        let (detector, _) = detector.finish();
        let once = detector.clean(&article(1));
        let twice = detector.clean(&once);
        assert_eq!(once, twice, "cleaning a cleaned document changes nothing");
    }

    #[test]
    fn blank_lines_are_not_furniture() {
        // Every document shares them, and removing them would join paragraphs
        // that the source separated.
        let mut detector = Boilerplate::new();
        for n in 0..MIN_DOCUMENTS_PER_HOST {
            detector.observe(
                Some("blogfa.com"),
                &format!("\n\n{}\n\n\nمتن {n}\n\n", article(n as usize)),
            );
        }
        let (detector, _) = detector.finish();
        assert!(!detector.is_removed(""));
    }
}
