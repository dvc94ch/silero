use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct Decoder {
    labels: Vec<String>,
    blank_idx: usize,
    two_idx: usize,
}

impl Decoder {
    pub fn new(labels: Vec<String>) -> Result<Self> {
        let blank_idx = labels
            .iter()
            .position(|label| label == "_")
            .context("invalid labels")?;
        let two_idx = labels
            .iter()
            .position(|label| label == "2")
            .context("invalid labels")?;
        Ok(Self {
            labels,
            blank_idx,
            two_idx,
        })
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Self::new(serde_json::from_str(json)?)
    }

    #[allow(unused)]
    pub fn from_path(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let labels: Vec<String> = serde_json::from_reader(reader)?;
        Self::new(labels)
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn decode(&self, argm: &[usize]) -> Result<String> {
        let mut pieces = vec![];
        for i in argm.iter().copied() {
            if i == self.two_idx {
                if pieces.is_empty() {
                    pieces.push(" ");
                } else {
                    pieces.push("$");
                    let last = pieces[pieces.len() - 2];
                    pieces.push(last);
                }
            } else if i != self.blank_idx {
                pieces.push(&self.labels[i]);
            }
        }
        let mut s = String::new();
        let mut last = None;
        for piece in pieces {
            let curr = Some(piece);
            if curr == last {
                continue;
            }
            last = curr;
            s.push_str(piece);
        }
        Ok(s.replace('$', "").trim().to_string())
    }
}
