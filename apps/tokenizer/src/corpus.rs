//! Bounded-memory streaming over UTF-8 corpus files.
//!
//! `Tokenizer::train` consumes a plain iterator of strings and cannot surface a
//! `Result`, so this adapter streams one line at a time, stops at the first I/O
//! failure, and lets the caller collect that failure afterwards. Memory use is
//! one line, not one corpus.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::path::PathBuf;

use thiserror::Error;

/// Failure while opening or reading a corpus file.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CorpusError {
    /// A corpus file could not be opened.
    #[error("failed to open corpus file `{path}`")]
    Open {
        /// Path that was opened.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// A corpus file could not be read to the end.
    #[error("failed while reading corpus file `{path}`")]
    Read {
        /// Path that was being read.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// No corpus file was supplied.
    #[error("at least one corpus file is required")]
    Empty,
}

/// Lines from several files, in the order the files were given.
pub struct CorpusLines {
    readers: VecDeque<(PathBuf, Lines<BufReader<File>>)>,
    failure: Option<(PathBuf, std::io::Error)>,
}

impl CorpusLines {
    /// Opens every file up front, so a missing corpus fails before training.
    pub fn open(paths: &[PathBuf]) -> Result<Self, CorpusError> {
        if paths.is_empty() {
            return Err(CorpusError::Empty);
        }

        let mut readers = VecDeque::with_capacity(paths.len());
        for path in paths {
            let file = File::open(path).map_err(|source| CorpusError::Open {
                path: path.clone(),
                source,
            })?;
            readers.push_back((path.clone(), BufReader::new(file).lines()));
        }

        Ok(Self {
            readers,
            failure: None,
        })
    }

    /// Takes the I/O failure that stopped iteration, if there was one.
    pub fn take_failure(&mut self) -> Option<CorpusError> {
        self.failure
            .take()
            .map(|(path, source)| CorpusError::Read { path, source })
    }
}

impl Iterator for CorpusLines {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            let front = self.readers.front_mut()?;
            let path = front.0.clone();
            let next = front.1.next();

            match next {
                Some(Ok(line)) => return Some(line),
                Some(Err(source)) => {
                    self.failure = Some((path, source));
                    self.readers.clear();
                    return None;
                }
                None => {
                    self.readers.pop_front();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{CorpusError, CorpusLines};

    /// Scratch directory inside the workspace, so tests never depend on the
    /// system temporary directory being writable.
    fn scratch_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("tmp")
            .join("tokenizer-cli-corpus")
    }

    fn fixture(name: &str) -> PathBuf {
        let directory = scratch_dir();
        std::fs::create_dir_all(&directory).expect("scratch directory");
        let path = directory.join(name);
        std::fs::write(&path, "first\nsecond\n").expect("fixture write");
        path
    }

    #[test]
    fn lines_are_streamed_in_file_order() {
        let a = fixture("a.txt");
        let b = fixture("b.txt");

        let lines: Vec<String> = CorpusLines::open(&[a, b])
            .expect("both files exist")
            .collect();

        assert_eq!(lines, ["first", "second", "first", "second"]);
    }

    #[test]
    fn no_failure_is_recorded_for_a_clean_run() {
        let mut corpus = CorpusLines::open(&[fixture("clean.txt")]).expect("file exists");
        let _: Vec<String> = corpus.by_ref().collect();
        assert!(corpus.take_failure().is_none());
    }

    #[test]
    fn an_empty_path_list_is_rejected() {
        assert!(matches!(CorpusLines::open(&[]), Err(CorpusError::Empty)));
    }

    #[test]
    fn a_missing_file_fails_at_open_time_with_its_path() {
        let missing = scratch_dir().join("absent.txt");
        match CorpusLines::open(std::slice::from_ref(&missing)).err() {
            Some(CorpusError::Open { path, .. }) => assert_eq!(path, missing),
            other => panic!("expected an open failure, got {other:?}"),
        }
    }
}
