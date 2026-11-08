//! Cleaned features (Cadence, 2026-11-07), for score following.
//!
//! The pitch-class engine's cleaning, exposed as plain 48-note energies per frame: the
//! harmonic part of each frame (pitch_class::harmonic_note_energies — median filters,
//! soft mask, modified autocorrelation, no octave fix, no top-k), then the drone removed
//! (each note minus its median over a sliding stretch). Streamed in chunks so a long
//! recording never holds its magnitude spectra: the harmonic part of a frame only reads
//! PC_HPSS_TIME_FRAMES / 2 frames on each side, so chunks overlapping by that much give
//! exactly what one pass over the whole would.
use super::pitch_class::harmonic_note_energies;
use super::types::{Frame, InterpInds};
use crate::ff_config;
use rustfft::algorithm::Radix4;

const CHUNK: usize = 512;
const N: usize = ff_config::MIDI_NUM as usize;

pub struct CleanStream {
    mags: Vec<Vec<f32>>,
    started: bool,
    out: Vec<Frame>,
}

impl CleanStream {
    pub fn new() -> CleanStream {
        CleanStream { mags: Vec::new(), started: false, out: Vec::new() }
    }

    pub fn push(&mut self, mag: Vec<f32>, fft: &Radix4<f32>, interp: &InterpInds) {
        self.mags.push(mag);
        if self.mags.len() >= CHUNK + 2 * (ff_config::PC_HPSS_TIME_FRAMES / 2) {
            self.process(fft, interp, false);
        }
    }

    fn process(&mut self, fft: &Radix4<f32>, interp: &InterpInds, last: bool) {
        let kt = ff_config::PC_HPSS_TIME_FRAMES / 2;
        let n = self.mags.len();
        if n == 0 {
            return;
        }
        let fr = harmonic_note_energies(&self.mags, fft, interp);
        let from = if self.started { kt.min(n) } else { 0 };
        let to = if last { n } else { n - kt };
        if to > from {
            self.out.extend_from_slice(&fr[from..to]);
        }
        self.started = true;
        if last {
            self.mags.clear();
        } else {
            self.mags.drain(0..n - 2 * kt);
        }
    }

    /// Every frame fed since the last call, cleaned; resets the stream.
    pub fn finish(&mut self, fft: &Radix4<f32>, interp: &InterpInds) -> Vec<Frame> {
        self.process(fft, interp, true);
        self.started = false;
        std::mem::take(&mut self.out)
    }

    pub fn clear(&mut self) {
        self.mags.clear();
        self.started = false;
        self.out.clear();
    }
}

/// Keeps the `k` strongest of a frame's 48 energies and zeroes the rest; on a tie the lower note
/// is kept, as in the bench that measured it.
pub fn keep_strongest(frame: &mut Frame, k: usize) {
    if k == 0 || k >= N {
        return;
    }
    let mut order: Vec<usize> = (0..N).collect();
    order.sort_by(|&a, &b| frame[b].partial_cmp(&frame[a]).unwrap());
    for i in &order[k..] {
        frame[*i] = 0.;
    }
}

/// Each note minus its median over `span` frames: centred on the frame, or the `span`
/// frames up to it (`causal`, what a live reader can know). The median is taken once per
/// `stride` frames and held in between. `span` 0: nothing removed.
pub fn remove_drone_sliding(frames: &mut [Frame], span: usize, causal: bool, stride: usize) {
    let n = frames.len();
    if n == 0 || span == 0 {
        return;
    }
    let stride = stride.max(1);
    let mut col: Vec<f32> = Vec::with_capacity(span + 1);
    let mut meds = vec![0f32; ((n + stride - 1) / stride) * N];
    for b in 0..(n + stride - 1) / stride {
        let t = b * stride;
        let (lo, hi) = if causal { (t.saturating_sub(span), t + 1) } else { (t.saturating_sub(span / 2), (t + span / 2 + 1).min(n)) };
        for j in 0..N {
            col.clear();
            for f in &frames[lo..hi] {
                col.push(f[j]);
            }
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            meds[b * N + j] = col[col.len() / 2];
        }
    }
    for (t, f) in frames.iter_mut().enumerate() {
        let b = t / stride;
        for j in 0..N {
            f[j] = (f[j] - ff_config::PC_DRONE_FRACTION * meds[b * N + j]).max(0.);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::interpolate::compute_interp_inds;
    use rustfft::FftDirection;

    #[test]
    fn chunked_stream_equals_one_pass() {
        let fft = Radix4::new(ff_config::SPEC_WINDOW_SIZE, FftDirection::Forward);
        let interp = compute_interp_inds(&48000);
        let mut seed: u32 = 12345;
        let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 8) as f32 / 16777216.0 };
        let mags: Vec<Vec<f32>> = (0..1300).map(|_| (0..ff_config::SPEC_WINDOW_SIZE).map(|_| rnd()).collect()).collect();
        let whole = harmonic_note_energies(&mags, &fft, &interp);
        let mut s = CleanStream::new();
        for m in &mags { s.push(m.clone(), &fft, &interp); }
        let streamed = s.finish(&fft, &interp);
        assert_eq!(whole.len(), streamed.len());
        for (a, b) in whole.iter().zip(streamed.iter()) { assert_eq!(a, b); }
    }
}
