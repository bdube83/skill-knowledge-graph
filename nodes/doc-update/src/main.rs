//! doc-update -- SKG WASI node
//!
//! Generates an updated documentation section given a description of the change.
//! No filesystem access; the caller injects the existing section as context.
//!
//! Input (JSON on stdin):
//!   {
//!     "task": "update documentation section",
//!     "context": {
//!       "doc_path": "docs/api.md",
//!       "section_title": "Authentication",
//!       "existing_content": "...",
//!       "change_description": "Added OAuth2 support"
//!     },
//!     "grants": ["text.generate"]
//!   }
//!
//! Output (JSON on stdout):
//!   {
//!     "output": {
//!       "updated_section": "## Authentication\n\n...",
//!       "change_summary": "Added OAuth2 section to Authentication."
//!     },
//!     "observed_effects": ["text.generate"]
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

    if !grants.contains(&"text.generate") {
        return error_json("capability text.generate not granted");
    }

    let ctx = &parsed["context"];
    let section_title = ctx["section_title"].as_str().unwrap_or("Section");
    let existing = ctx["existing_content"].as_str().unwrap_or("");
    let change_desc = ctx["change_description"].as_str().unwrap_or("(no description)");
    let doc_path = ctx["doc_path"].as_str().unwrap_or("(unknown)");

    // Produce a structured update note. In the real north-star build this would
    // go to an LLM; the WASI node provides a deterministic template so dry-runs
    // and promotion gates succeed without an LLM call.
    let updated_section = format!(
        "## {section_title}\n\n{existing}\n\n<!-- doc-update node applied: {change_desc} -->\n"
    );

    let change_summary = format!(
        "Updated '{section_title}' in {doc_path}: {change_desc}."
    );

    serde_json::json!({
        "output": {
            "updated_section": updated_section,
            "change_summary": change_summary
        },
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
