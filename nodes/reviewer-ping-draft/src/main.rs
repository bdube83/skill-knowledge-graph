//! reviewer-ping-draft — SKG example WASI node
//!
//! Reads a JSON task context from stdin, writes a JSON result to stdout.
//! No network, no filesystem, no secrets. Effects: text.generate only.
//!
//! Input (JSON on stdin):
//!   { "task": "...", "context": { "pr_number": 42, "repo": "...",
//!     "author": "...", "reviewers": ["..."] }, "grants": ["text.generate"] }
//!
//! Output (JSON on stdout):
//!   { "output": { "message": "..." }, "observed_effects": ["text.generate"] }

use std::io::{self, Read, Write};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let result = process(&input);

    io::stdout().write_all(result.as_bytes()).unwrap();
}

fn process(input: &str) -> String {
    // Parse grants — abort if text.generate is not granted.
    let parsed: serde_json::Value = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("invalid input JSON: {e}")),
    };

    let grants = parsed["grants"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    if !grants.contains(&"text.generate") {
        return error_json("capability text.generate not granted");
    }

    let ctx = &parsed["context"];
    let pr_number = ctx["pr_number"].as_u64().unwrap_or(0);
    let repo = ctx["repo"].as_str().unwrap_or("(unknown)");
    let author = ctx["author"].as_str().unwrap_or("(unknown)");
    let reviewers: Vec<&str> = ctx["reviewers"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let reviewer_mentions = if reviewers.is_empty() {
        "team".to_string()
    } else {
        reviewers
            .iter()
            .map(|r| format!("@{r}"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let message = format!(
        "Hi {reviewer_mentions}, could you review PR #{pr_number} in `{repo}`? \
         It was opened by @{author}. Let me know if you need any context."
    );

    serde_json::json!({
        "output": { "message": message },
        "observed_effects": ["text.generate"]
    })
    .to_string()
}

fn error_json(msg: &str) -> String {
    serde_json::json!({
        "error": msg,
        "output": {},
        "observed_effects": []
    })
    .to_string()
}
