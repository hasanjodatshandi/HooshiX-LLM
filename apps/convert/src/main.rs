//! `convert`: turn a downloaded dataset file into the form the pipeline reads.
//!
//! # Why this exists
//!
//! FineWeb2 publishes Parquet, and FineWeb2's `fas_Arab` split is the Persian
//! source the bilingual plan names. The pipeline reads plain text and JSON lines.
//! Something has to bridge the two, and it is deliberately not `llm-data`: a
//! Parquet reader brings an Arrow dependency tree, and the crate the model and
//! trainer link against should not carry it.
//!
//! # What it does, and what it deliberately does not
//!
//! It reads one Parquet file and writes one JSON object per line, with the text
//! under a named field. Nothing else: no filtering, no deduplication, no
//! normalization. Those belong to the pipeline, which already does them and has
//! tests for them, and doing them twice would mean two places to look when a
//! count is wrong.
//!
//! # The honest cost
//!
//! This produces an intermediate file. The Persian shard is 4.6 GB of Parquet
//! and becomes several GB of JSON lines, so a full Stage-1 run would hold both.
//! That is the price of keeping the core crates clean, and it is worth paying
//! once. If Parquet becomes a primary format rather than a source format, the
//! right answer changes to reading it in `llm-data` directly, and this tool
//! becomes unnecessary.

#![deny(unsafe_code)]

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// The input could not be read or the output could not be written.
const EXIT_IO: u8 = 2;
/// The Parquet file could not be decoded.
const EXIT_PARQUET: u8 = 3;
/// The file does not hold the column that was asked for.
const EXIT_NO_COLUMN: u8 = 4;
/// A cleaning adapter could not be read or parsed.
const EXIT_CONFIG: u8 = 5;

#[derive(Debug, Parser)]
#[command(
    name = "convert",
    about = "Turn a downloaded dataset file into one JSON object per line"
)]
struct Arguments {
    /// Parquet file to read.
    #[arg(long)]
    input: PathBuf,

    /// JSON lines file to write.
    #[arg(long)]
    output: PathBuf,

    /// Name of the column holding the text.
    #[arg(long, default_value = "text")]
    column: String,

    /// Stop after this many rows. Useful for taking a sample without reading
    /// the whole file.
    #[arg(long)]
    limit: Option<usize>,

    /// Row group size to read at a time. Larger is faster and holds more memory.
    #[arg(long, default_value_t = 8192)]
    batch: usize,

    /// Print the file's schema and exit.
    ///
    /// Exists because a dataset that ships its own `language_score`,
    /// `quality_score` or `minhash_cluster_size` columns has already done work
    /// this pipeline would otherwise repeat, and the only way to know is to look.
    #[arg(long)]
    schema: bool,

    /// Per-language cleaning adapter, whose `[source]` section is applied here.
    ///
    /// The dataset ran a language model over every document and a global MinHash
    /// over the whole corpus. Reading those columns is not laziness: recomputing
    /// either would produce a second answer to a question already answered, from
    /// a weaker method. What this project adds is the part the dataset does not
    /// have, and that part starts in `llm-data`.
    #[arg(long)]
    adapter: Option<PathBuf>,

    /// Write plain text, one document per line, instead of JSON.
    ///
    /// The tokenizer trainer consumes UTF-8 text and learns merges over whatever
    /// it is given. Handed JSON lines it would learn `{"text": "` as a token,
    /// because that prefix appears once per document — spending vocabulary on
    /// structural noise and distorting every token count afterwards.
    #[arg(long, value_name = "PATH")]
    plain_text: Option<PathBuf>,
}

/// The `[source]` section of a per-language adapter.
///
/// Only the fields this tool uses are named. The adapter carries much more — the
/// per-script heuristics, the stopword list, the normalization switches — and it
/// is deliberately not an error to carry keys a particular consumer ignores,
/// because the same file is read by two consumers with different jobs.
#[derive(Debug, serde::Deserialize)]
struct Adapter {
    language: String,
    source: SourceThresholds,
}

#[derive(Debug, serde::Deserialize)]
struct SourceThresholds {
    min_language_score: f64,
    max_minhash_cluster_size: i64,
}

impl Adapter {
    fn load(path: &Path) -> Result<Self, (u8, String)> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| (EXIT_IO, format!("{}: {error}", path.display())))?;
        toml::from_str(&text).map_err(|error| (EXIT_CONFIG, format!("{}: {error}", path.display())))
    }
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();

    // A JSON lines input is not Parquet and is handled by its own path. The
    // extension decides, because that is what a caller has to go on and a wrong
    // guess fails immediately rather than producing a wrong corpus.
    //
    // `.jsonl.gz` is matched as a whole rather than by extension, because that
    // name's extension is `gz` and its content is still JSON lines.
    let name = arguments
        .input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let is_jsonl = name.ends_with(".jsonl") || name.ends_with(".jsonl.gz");

    if is_jsonl {
        return match extract_text(&arguments) {
            Ok(written) => {
                println!(
                    "convert: wrote {written} documents to {}",
                    arguments
                        .plain_text
                        .as_ref()
                        .unwrap_or(&arguments.output)
                        .display()
                );
                ExitCode::SUCCESS
            }
            Err((code, message)) => {
                eprintln!("convert: {message}");
                ExitCode::from(code)
            }
        };
    }

    if arguments.schema {
        return match print_schema(&arguments.input) {
            Ok(()) => ExitCode::SUCCESS,
            Err((code, message)) => {
                eprintln!("convert: {message}");
                ExitCode::from(code)
            }
        };
    }

    match run(&arguments) {
        Ok(written) => {
            println!(
                "convert: wrote {written} documents to {}",
                arguments.output.display()
            );
            ExitCode::SUCCESS
        }
        Err((code, message)) => {
            eprintln!("convert: {message}");
            ExitCode::from(code)
        }
    }
}

/// Prints every column and its Arrow type, then the first row group's size.
///
/// A dataset's own scores are worth more than anything this project can compute
/// from the text, because they come from the pipeline that produced the data and
/// from models trained for the language in question. Reading the schema first is
/// how that is discovered rather than assumed.
fn print_schema(input: &Path) -> Result<(), (u8, String)> {
    let file =
        File::open(input).map_err(|error| (EXIT_IO, format!("{}: {error}", input.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| (EXIT_PARQUET, format!("{}: {error}", input.display())))?;

    let schema = builder.schema();
    let metadata = builder.metadata();

    println!("{}", input.display());
    println!("  rows          {}", metadata.file_metadata().num_rows());
    println!("  row groups    {}", metadata.num_row_groups());
    println!("  columns       {}", schema.fields().len());
    for field in schema.fields() {
        println!("    {:<24} {}", field.name(), field.data_type());
    }
    Ok(())
}

/// Reads JSON lines and writes the text field of each, one document per line.
///
/// Used for the Common Pile shards, which are already JSON lines with a `text`
/// field, and for producing tokenizer training text from anything this tool has
/// converted. A line that does not parse is counted and skipped rather than
/// ending the run: a shard of a million documents should not be lost to one bad
/// record, and the count says whether it was one or many.
fn extract_text(arguments: &Arguments) -> Result<u64, (u8, String)> {
    let source = File::open(&arguments.input)
        .map_err(|error| (EXIT_IO, format!("{}: {error}", arguments.input.display())))?;

    // Common Pile ships raw gzip streams rather than tar archives, which is why
    // `tar` refuses them. Decoding here also means the decompressed shard never
    // has to exist: 950 MB of intermediate JSON lines is disk this project does
    // not spend, and the pass reads, converts and discards in one go.
    let reader: Box<dyn BufRead> = if arguments
        .input
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gz"))
    {
        Box::new(BufReader::with_capacity(
            1 << 20,
            flate2::read::MultiGzDecoder::new(source),
        ))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, source))
    };

    let destination = arguments.plain_text.as_ref().unwrap_or(&arguments.output);
    let output = File::create(destination)
        .map_err(|error| (EXIT_IO, format!("{}: {error}", destination.display())))?;
    let mut writer = BufWriter::with_capacity(1 << 20, output);

    let mut written: u64 = 0;
    let mut unparsed: u64 = 0;
    let mut without_text: u64 = 0;

    for line in reader.lines() {
        let line = line.map_err(|error| (EXIT_IO, format!("reading: {error}")))?;
        if line.trim().is_empty() {
            continue;
        }
        if arguments.limit.is_some_and(|limit| written >= limit as u64) {
            break;
        }

        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            unparsed += 1;
            continue;
        };
        let Some(text) = value
            .get(&arguments.column)
            .and_then(|field| field.as_str())
        else {
            without_text += 1;
            continue;
        };

        // A document's own newlines are not preserved as line breaks, because
        // the output is line-oriented and a line break inside a document would
        // split it in two for every reader downstream. Paragraph structure is
        // kept as a single space, which is what the tokenizer sees anyway.
        writeln!(writer, "{}", text.replace('\n', " "))
            .map_err(|error| (EXIT_IO, format!("writing: {error}")))?;
        written += 1;
    }

    writer
        .flush()
        .map_err(|error| (EXIT_IO, format!("flushing: {error}")))?;

    if unparsed > 0 || without_text > 0 {
        println!(
            "convert: {unparsed} line(s) did not parse as JSON, {without_text} had no `{}` field",
            arguments.column
        );
    }

    Ok(written)
}

fn run(arguments: &Arguments) -> Result<u64, (u8, String)> {
    let adapter = match &arguments.adapter {
        Some(path) => Some(Adapter::load(path)?),
        None => None,
    };
    let file = File::open(&arguments.input)
        .map_err(|error| (EXIT_IO, format!("{}: {error}", arguments.input.display())))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| {
            (
                EXIT_PARQUET,
                format!("{}: {error}", arguments.input.display()),
            )
        })?
        .with_batch_size(arguments.batch);

    // The column is looked up once, from the file's own schema, so a wrong name
    // fails before a single row is written rather than silently producing an
    // empty file.
    let schema = builder.schema().clone();
    let column_index = schema.index_of(&arguments.column).map_err(|_| {
        let available: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        (
            EXIT_NO_COLUMN,
            format!(
                "no column `{}` in {}; the file has: {}",
                arguments.column,
                arguments.input.display(),
                available.join(", ")
            ),
        )
    })?;

    let reader = builder.build().map_err(|error| {
        (
            EXIT_PARQUET,
            format!("{}: {error}", arguments.input.display()),
        )
    })?;

    // One destination, decided once: plain text when the tokenizer path asked
    // for it, JSON lines otherwise. Deciding it here rather than inside the loop
    // keeps the write itself to a single branch.
    let destination = arguments.plain_text.as_ref().unwrap_or(&arguments.output);
    let output = File::create(destination)
        .map_err(|error| (EXIT_IO, format!("{}: {error}", destination.display())))?;
    let mut writer = BufWriter::with_capacity(1 << 20, output);
    let plain = arguments.plain_text.is_some();

    let mut written: u64 = 0;
    let mut dropped_language_score: u64 = 0;
    let mut dropped_cluster: u64 = 0;
    let mut lacked_metadata: u64 = 0;

    // The threshold columns are looked up by name once, so a file that lacks them
    // is reported rather than silently letting every row through unfiltered.
    let score_index = arguments
        .adapter
        .as_ref()
        .and_then(|_| schema.index_of("language_score").ok());
    let cluster_index = arguments
        .adapter
        .as_ref()
        .and_then(|_| schema.index_of("minhash_cluster_size").ok());

    if let Some(adapter) = &adapter {
        println!(
            "convert: adapter `{}`, language {}, min_language_score {}, max_minhash_cluster_size {}",
            arguments
                .adapter
                .as_ref()
                .map_or_else(String::new, |p| p.display().to_string()),
            adapter.language,
            adapter.source.min_language_score,
            adapter.source.max_minhash_cluster_size
        );
    }

    for batch in reader {
        let batch: RecordBatch =
            batch.map_err(|error| (EXIT_PARQUET, format!("reading a row group: {error}")))?;
        let rows = batch.num_rows();

        for row in 0..rows {
            if arguments.limit.is_some_and(|limit| written >= limit as u64) {
                break;
            }
            let text = text_at(&batch, column_index, row).map_err(|m| (EXIT_PARQUET, m))?;

            // The dataset's own verdicts are applied before anything this project
            // computes. A drop here is counted under the dataset's reason, not
            // mixed in with ours, so the two are never confused in a report.
            if let Some(adapter) = &adapter {
                match score_index.and_then(|i| float_at(&batch, i, row)) {
                    Some(score) if score < adapter.source.min_language_score => {
                        dropped_language_score += 1;
                        continue;
                    }
                    None => lacked_metadata += 1,
                    _ => {}
                }
                match cluster_index.and_then(|i| int_at(&batch, i, row)) {
                    Some(size) if size > adapter.source.max_minhash_cluster_size => {
                        dropped_cluster += 1;
                        continue;
                    }
                    None => lacked_metadata += 1,
                    _ => {}
                }
            }

            // Every column is carried through, not only the text. The dataset's
            // own url, language, score and cluster size are the inputs the
            // per-language rules need, and a converter that wrote `{"text": …}`
            // would throw away the most reliable metadata in the corpus.
            //
            // Unless plain text was asked for, in which case only the text is
            // written and the metadata is dropped here — which is why the
            // tokenizer path is the only caller that asks for it.
            if plain {
                writeln!(writer, "{}", text.replace('\n', " "))
                    .map_err(|error| (EXIT_IO, format!("writing: {error}")))?;
                written += 1;
                continue;
            }

            let mut object = serde_json::Map::new();
            object.insert("text".to_owned(), serde_json::Value::String(text));
            for (index, field) in schema.fields().iter().enumerate() {
                if index == column_index {
                    continue;
                }
                if let Some(value) = value_at(&batch, index, row) {
                    object.insert(field.name().clone(), value);
                }
            }

            let line = serde_json::Value::Object(object);
            writeln!(writer, "{line}").map_err(|error| (EXIT_IO, format!("writing: {error}")))?;
            written += 1;
        }

        if arguments.limit.is_some_and(|limit| written >= limit as u64) {
            break;
        }
    }

    writer
        .flush()
        .map_err(|error| (EXIT_IO, format!("flushing: {error}")))?;

    if adapter.is_some() {
        println!(
            "convert: kept {written}, dropped {dropped_language_score} below language_score, \
             {dropped_cluster} above minhash_cluster_size, {lacked_metadata} missing a score"
        );
    }

    Ok(written)
}

/// Reads one row's value from a column, as JSON, for every type this expects.
///
/// `None` for a null, an unexpected type, or an index past the end. The caller
/// distinguishes "absent" from "present and failing a threshold", because a row
/// with no language score has not been judged and must not be treated as judged.
fn value_at(batch: &RecordBatch, index: usize, row: usize) -> Option<serde_json::Value> {
    use arrow_array::{Float64Array, Int64Array};

    let column = batch.column(index);
    if column.is_null(row) {
        return None;
    }
    if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
        return Some(serde_json::Value::String(array.value(row).to_owned()));
    }
    if let Some(array) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Some(serde_json::Value::String(array.value(row).to_owned()));
    }
    if let Some(array) = column.as_any().downcast_ref::<Float64Array>() {
        return serde_json::Number::from_f64(array.value(row)).map(serde_json::Value::Number);
    }
    if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
        return Some(serde_json::Value::Number(array.value(row).into()));
    }
    None
}

fn text_at(batch: &RecordBatch, index: usize, row: usize) -> Result<String, String> {
    match value_at(batch, index, row) {
        Some(serde_json::Value::String(text)) => Ok(text),
        Some(_) => Err("the text column is not a string".to_owned()),
        None => Ok(String::new()),
    }
}

fn float_at(batch: &RecordBatch, index: usize, row: usize) -> Option<f64> {
    match value_at(batch, index, row) {
        Some(serde_json::Value::Number(number)) => number.as_f64(),
        _ => None,
    }
}

fn int_at(batch: &RecordBatch, index: usize, row: usize) -> Option<i64> {
    match value_at(batch, index, row) {
        Some(serde_json::Value::Number(number)) => number.as_i64(),
        _ => None,
    }
}

/// Resolved path, used only in messages so a relative argument is unambiguous.
#[allow(dead_code)]
fn display(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}
