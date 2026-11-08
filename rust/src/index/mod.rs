use serde_json;
pub mod schema;

use serde::{Deserialize, Serialize};
use schema::{SettingID, TuneAliases, TuneSettings};


#[derive(Serialize, Deserialize)] 
pub struct TuneIndex {
    pub settings: TuneSettings,
    pub aliases: TuneAliases,
    /// The pitch-class n-gram codes of every setting (Cadence, 2026-11-08), built once
    /// the index is folded (QueryEngine), read by the any-key n-gram pass. Empty otherwise.
    #[serde(skip)]
    pub grams: PitchClassGrams,
}

/// Every setting's 4-symbol grams as the any-key pass reads them, computed once rather than
/// rolled anew from the contour on every query: setting `i` is `ids[i]`, the gram ending at
/// each of its symbols from the fourth on `codes[starts[i]..starts[i + 1]]` — four bits a
/// pitch class, first symbol highest; `NO_GRAM` where the four symbols are not all pitch
/// classes (none in a folded index), which no query gram can equal. One block, read as a
/// stream. Order is the map's: no ranking depends on it (every one breaks ties by id).
#[derive(Default)]
pub struct PitchClassGrams {
    pub ids: Vec<SettingID>,
    pub starts: Vec<u32>,
    pub codes: Vec<u16>,
}

pub const NO_GRAM: u16 = u16::MAX;

impl TuneIndex {
    pub fn build_pitch_class_grams(&mut self) {
        let total: usize = self.settings.values().map(|s| s.contour.len().saturating_sub(3)).sum();
        let mut g = PitchClassGrams {
            ids: Vec::with_capacity(self.settings.len()),
            starts: Vec::with_capacity(self.settings.len() + 1),
            codes: Vec::with_capacity(total),
        };
        for (id, setting) in &self.settings {
            g.ids.push(id.clone());
            g.starts.push(g.codes.len() as u32);
            let mut code: u16 = 0;
            let mut run = 0usize;
            for (k, &ch) in setting.contour.as_bytes().iter().enumerate() {
                let v = ch.wrapping_sub(b'a');
                if v >= 12 {
                    run = 0;
                } else {
                    code = (code << 4) | v as u16;
                    run += 1;
                }
                if k >= 3 {
                    g.codes.push(if run >= 4 { code } else { NO_GRAM });
                }
            }
        }
        g.starts.push(g.codes.len() as u32);
        self.grams = g;
    }
}

pub fn tune_index_from_string(json_string: &String) -> TuneIndex {
    return serde_json::from_str(json_string).expect("Couldn't parse index");
}
