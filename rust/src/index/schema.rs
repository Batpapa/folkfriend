use serde::{Deserialize, Serialize};
use std::collections;

pub type TuneSettings = collections::HashMap<SettingID, Setting>;
pub type TuneAliases = collections::HashMap<TuneID, Vec<SettingID>>;

// These are very deliberately strings of integers.
//   Otherwise rust / wasm / serde / etc. problems.
pub type TuneID = String;
pub type SettingID = String;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Setting {
    pub tune_id: TuneID,
    pub meter: String,
    pub mode: String,
    pub abc: String,
    pub dance: String,
    pub contour: String,
    /// Which index the setting comes from, when the caller merged several (Cadence,
    /// 2026-11-16): "thesession", "cards"… — named by the caller, read only by
    /// QueryEngine::set_source_order. Absent in an index of one source. An Option,
    /// not a String with a default: serde_wasm_bindgen 0.3 hands a missing property
    /// over as `undefined`, which a default does not catch (load panicked) and an
    /// Option reads as None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// The source's place in that order, set by QueryEngine::rank_sources.
    #[serde(skip)]
    pub tie_rank: u8,
}
