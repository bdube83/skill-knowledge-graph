//! git-summary -- SKG WASI node
//!
//! Produces a structured summary of recent git activity for a given repo/branch.
//! No actual git subprocess is run inside WASM; the caller injects the log lines
//! as context. The node formats and summarises them.
//!
//! Input (JSON on stdin):
//!   {
//!     "task": "summarise recent git activity",
//!     "context": {
//!       "repo": "owner/repo",
//!       "branch": "main",
//!       "commits": [
//!         { "sha": "abc1234", "author": "alice", "message": "fix: auth bug" }
//!       ]
//!     },
//!     "grants": ["git.read"]
//!   }
//!
//! Output (JSON on stdout):
//!   {
//!     "output": {
//!       "summary": "3 commits on main since last week. ...",
//!       "authors": ["alice", "bob"],
//!       "commit_count": 3
//!     },
//!     "observed_effects": ["git.read"]
//!   }

use std::io::{self, Read, Write};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let result = process(&input);
    io::stdout().write_all(result.as_bytes()).unwrap();
}

fn process(input: &str) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(e) => return error_json(&format!("invalid input JSON: {e}")),
    };

    let grants = parsed["grants"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    if !grants.contains(&"git.read") {
        return error_json("capability git.read not granted");
    }

    let ctx = &parsed["context"];
    let repo = ctx["repo"].as_str().unwrap_or("(unknown)");
    let branch = ctx["branch"].as_str().unwrap_or("main");

    let commits = ctx["commits"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
    let commit_count = commits.len();

    let mut authors: Vec<String> = commits
        .iter()
        .filter_map(|c| c["author"].as_str().map(|s| s.to_string()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    authors.sort();

    let messages: Vec<&str> = commits
        .iter()
        .filter_map(|c| c["message"].as_str())
        .take(5)
        .collect();

    let summary = if commit_count == 0 {
        format!("No recent commits on {branch} in {repo}.")
    } else {
        let author_list = if authors.is_empty() {
            "unknown authors".to_string()
        } else {
            authors.join(", ")
        };
        let msg_preview = messages.join("; ");
        format!(
            "{commit_count} commit{s} on {branch} in {repo} by {author_list}. Recent: {msg_preview}.",
            s = if commit_count == 1 { "" } else { "s" }
        )
    };

    serde_json::json!({
        "output": {
            "summary": summary,
            "authors": authors,
            "commit_count": commit_count
        },
        "observed_effects": ["git.read"]
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
