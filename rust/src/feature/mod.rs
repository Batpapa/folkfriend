pub mod autocorrelate;
pub mod clean;
pub mod interpolate;
pub mod normalise;
pub mod pitch_class;
pub mod signal;
pub mod types;
pub mod window;

use crate::ff_config;
use interpolate::compute_interp_inds;
use rustfft::{algorithm::Radix4, FftDirection};
use signal::validate_sample_rate;
use std::convert::TryInto;
use types::{Features, Frame, InterpInds, Window};
use window::gen_blackman_window;

pub struct FeatureExtractor {
    pub sample_rate: u32,
    pub window_function: Window,
    pub fft: Radix4<f32>,
    pub interp_inds: InterpInds,
    pub features: Features,
    /// Pitch-class mode (Cadence, 2026-11-07): frames are kept as magnitude
    /// spectra and turned into features per window (pitch_class.rs) instead of
    /// one 48-note frame at a time. `features` stays empty in this mode.
    pub pitch_class_mode: bool,
    pub magnitudes: Vec<Vec<f32>>,
    /// Pitch-class mode only: each window's tuning estimated and corrected
    /// (pitch_class::window_tuning) — set with the query's every-key search by
    /// FolkFriend::set_any_pitch.
    pub pitch_class_tuning: bool,
    /// RESEARCH (2026-11-07, clean.rs): frames streamed into cleaned 48-note energies.
    pub clean_mode: bool,
    pub clean: clean::CleanStream,
}

impl FeatureExtractor {
    pub fn new(sample_rate: u32) -> Result<FeatureExtractor, signal::SampleRateError> {
        if !validate_sample_rate(&sample_rate) {
            return Err(signal::SampleRateError);
        }

        Ok(FeatureExtractor {
            sample_rate: sample_rate,
            window_function: gen_blackman_window(),
            fft: Radix4::new(ff_config::SPEC_WINDOW_SIZE, FftDirection::Forward),
            interp_inds: compute_interp_inds(&sample_rate),
            features: Vec::new(),
            pitch_class_mode: false,
            magnitudes: Vec::new(),
            pitch_class_tuning: false,
            clean_mode: false,
            clean: clean::CleanStream::new(),
        })
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        // Need to know the sample rate to undertake the linear resampling step
        validate_sample_rate(&sample_rate);
        self.interp_inds = compute_interp_inds(&sample_rate);
    }

    pub fn feed_signal(&mut self, signal: Vec<f32>) {
        // Implicit floor division
        let num_windows = signal.len() / ff_config::SPEC_WINDOW_SIZE;
        let last_ind = num_windows * ff_config::SPEC_WINDOW_SIZE;
        for i in (0..last_ind).step_by(ff_config::SPEC_WINDOW_SIZE) {
            self.feed_window(
                signal[i..i + ff_config::SPEC_WINDOW_SIZE]
                    .try_into()
                    .expect("Invalid window length"),
            );
        }
    }
    pub fn feed_window(&mut self, window: [f32; ff_config::SPEC_WINDOW_SIZE]) {
        if self.clean_mode {
            let mag = pitch_class::magnitude_spectrum(&window, &self.window_function, &self.fft);
            self.clean.push(mag, &self.fft, &self.interp_inds);
            return;
        }
        if self.pitch_class_mode {
            self.magnitudes.push(pitch_class::magnitude_spectrum(&window, &self.window_function, &self.fft));
            return;
        }
        let frame: Frame = autocorrelate::modified_autocorrelation(
            window,
            &self.window_function,
            &self.fft,
            &self.interp_inds,
        );
        self.features.push(frame);
    }

    pub fn flush(&mut self) {
        self.features = Vec::new();
        self.magnitudes = Vec::new();
        self.clean.clear();
    }

    /// The window fed so far as pitch-class features (pitch-class mode only), and the
    /// tuning it was read at when it is estimated (pitch_class_tuning).
    pub fn pitch_class_features(&self) -> (Features, Option<f32>) {
        pitch_class::features_from_magnitudes(&self.magnitudes, &self.fft, &self.interp_inds, if self.pitch_class_tuning { Some(self.sample_rate) } else { None })
    }
}
