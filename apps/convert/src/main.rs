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
use std::io::{BufWriter, Write};
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
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();

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

fn run(arguments: &Arguments) -> Result<u64, (u8, String)> {
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

    let output = File::create(&arguments.output)
        .map_err(|error| (EXIT_IO, format!("{}: {error}", arguments.output.display())))?;
    let mut writer = BufWriter::new(output);

    let mut written: u64 = 0;
    for batch in reader {
        let batch: RecordBatch =
            batch.map_err(|error| (EXIT_PARQUET, format!("reading a row group: {error}")))?;

        for text in texts(&batch, column_index).map_err(|message| (EXIT_PARQUET, message))? {
            if arguments.limit.is_some_and(|limit| written >= limit as u64) {
                break;
            }
            // A document that holds nothing is written anyway rather than
            // dropped here: the pipeline counts what it drops and why, and a
            // converter that quietly removes rows makes those counts wrong.
            let line = serde_json::json!({ "text": text });
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

    Ok(written)
}

/// Reads the text column of one batch, whichever string layout it uses.
///
/// Parquet files written by different producers use different Arrow types for
/// the same logical column — `utf8` or `large_utf8` are both common, and which
/// one appears depends on the writer's row-count assumptions rather than on the
/// data. Accepting both is not defensive coding; it is the difference between
/// reading FineWeb2 and failing on it.
fn texts(batch: &RecordBatch, index: usize) -> Result<Vec<String>, String> {
    let column = batch.column(index);

    if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
        return Ok((0..array.len())
            .map(|row| {
                if array.is_null(row) {
                    String::new()
                } else {
                    array.value(row).to_owned()
                }
            })
            .collect());
    }

    if let Some(array) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok((0..array.len())
            .map(|row| {
                if array.is_null(row) {
                    String::new()
                } else {
                    array.value(row).to_owned()
                }
            })
            .collect());
    }

    Err(format!(
        "the text column is `{:?}`, which is neither utf8 nor large_utf8",
        column.data_type()
    ))
}

/// Resolved path, used only in messages so a relative argument is unambiguous.
#[allow(dead_code)]
fn display(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}
