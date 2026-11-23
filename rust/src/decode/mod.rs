mod beam_search;
mod contour;
mod octave;
mod pitch_model;
pub mod types;

use crate::feature::types::{Features, Frame};
use crate::ff_config;
use crate::feature::signal;
use std::fmt;
use types::{Contour, ContourString, LatticePath};
pub use contour::{set_note_filter, set_tempo_model, set_tempo_range, ContourDebugFeatures, TempoCandidateScore};

pub struct FeatureDecoder {
    pub sample_rate: u32
}

#[derive(Debug, Clone)]
pub struct DecoderError;

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Error decoding features")
    }
}

impl FeatureDecoder {
    pub fn new(sample_rate: u32) -> Result<FeatureDecoder, signal::SampleRateError> {
        if !signal::validate_sample_rate(&sample_rate) {
            return Err(signal::SampleRateError);
        }
        
        Ok(FeatureDecoder {
            sample_rate: sample_rate
        })
    }

    pub fn decode_lattice_path(
        &self,
        features: &mut Features,
    ) -> Result<LatticePath, DecoderError> {
        beam_search::decode(features)
    }

    pub fn decode_contour(
        &self,
        lattice_path: &LatticePath,
        features: &Features
    ) -> Result<ContourString, DecoderError> {
        let mut contour: Contour = contour::contour_from_lattice_path(lattice_path, features, self.sample_rate)?;
        octave::correct_contour_octave(&mut contour);
        return Ok(types::contour_to_contour_string(&contour));
    }

    /// Pitch-class transcription (Cadence, 2026-11-07): features from
    /// feature::pitch_class, lattice over 12 classes, the default note/tempo
    /// stage. No octave correction: the contour has no octave. Its symbols are
    /// the first twelve ('a' = C ... 'l' = B), to be queried against an index
    /// folded the same way (QueryEngine::use_pitch_classes).
    pub fn decode_pitch_class_contour(
        &self,
        features: &mut Features
    ) -> Result<(ContourString, ContourDebugFeatures), DecoderError> {
        let lattice_path = beam_search::decode_pitch_classes(features)?;
        let (contour, debug) = contour::contour_from_lattice_path_debug(&lattice_path, features, self.sample_rate)?;
        return Ok((types::contour_to_contour_string(&contour), debug));
    }

    /// Transcription (Cadence, 2026-11-23): the pitch-class decoder over `classes`, each note
    /// returned with the energy of every octave of its class in `notes` (48-note frames, the
    /// same frames before the fold), summed over the note — for the octave to be chosen by the
    /// caller. Flat, 8 values a note: class (0 = C), first frame, frames, power (the class's
    /// mean energy as the decoder normalised it), then the 4 octaves' energies from MIDI_LOW up.
    pub fn decode_class_notes(&self, classes: &mut Features, notes: &[Frame]) -> Result<Vec<f32>, DecoderError> {
        const N: usize = ff_config::MIDI_NUM as usize;
        let path = beam_search::decode_pitch_classes(classes)?;
        let mut out = Vec::new();
        let mut i = 0;
        while i < path.len() {
            let c = path[i];
            let mut j = i;
            while j < path.len() && path[j] == c {
                j += 1;
            }
            let power: f32 = (i..j).map(|t| classes[t][c]).sum::<f32>() / (j - i) as f32;
            out.extend_from_slice(&[c as f32, i as f32, (j - i) as f32, power]);
            let mut o = c;
            while o < N {
                out.push((i..j).map(|t| notes[t][o]).sum());
                o += 12;
            }
            i = j;
        }
        Ok(out)
    }

    // Debug/experimental instrumentation (2026-08-18) — same computation as
    // `decode_contour`, additionally surfacing the note/tempo features that
    // are otherwise discarded. Never called from the production transcribe
    // path (`transcribe_pcm_buffer`); see `lib.rs`'s `transcribe_pcm_buffer_debug`.
    pub fn decode_contour_debug(
        &self,
        lattice_path: &LatticePath,
        features: &Features
    ) -> Result<(ContourString, ContourDebugFeatures), DecoderError> {
        let (mut contour, debug) = contour::contour_from_lattice_path_debug(lattice_path, features, self.sample_rate)?;
        octave::correct_contour_octave(&mut contour);
        return Ok((types::contour_to_contour_string(&contour), debug));
    }
}
