// Pitch-class features (Cadence, 2026-11-07).
//
// The default path turns every 1024-sample frame into 48 note energies on its own
// (`autocorrelate::modified_autocorrelation`) and decodes a melody over 48 pitches. This
// path keeps each frame's magnitude spectrum instead and, once the whole window is in,
// works on the window as a whole:
//
//   1. Harmonic/percussive separation of the magnitude spectra (median filters of
//      PC_HPSS_TIME_FRAMES frames along time and PC_HPSS_FREQ_BINS bins along frequency,
//      soft mask H^2 / (H^2 + P^2)). A sustained note is a horizontal line in the
//      spectrogram, a strum, a banjo attack or a bodhran hit a vertical one: the mask keeps
//      the first and removes the second before the autocorrelation.
//   2. The same modified autocorrelation as the default path (cube-root compression,
//      second FFT, interpolation onto 48 MIDI notes), without its per-frame octave fix
//      and top-5 — both made redundant by step 4.
//   3. Drone removal: each note loses PC_DRONE_FRACTION of its median energy over the
//      window. What does not change over 10 s (pipes' drones, held chords, hum, room
//      resonance) has a high median; a melody keeps changing note and has a low one.
//   4. Octaves summed into the 12 pitch classes, PC_RETAINED_CLASSES kept per frame.
//      Two instruments playing the same tune in different registers (whistle and banjo,
//      fiddle and low whistle) then add up instead of competing for the note, and the
//      autocorrelation's own octave ambiguity (a peak at twice the period) disappears.
//
// The decoder then follows a melody over 12 states (beam_search::decode_pitch_classes) and
// the index is folded the same way (QueryEngine::use_pitch_classes).
//
// Measured on Cadence's 17 annotated sessions (two-fold cross-validation of the detector's
// floor): 675/698 tunes for 16 false positives, against 645/698 for 27 with the default
// path; see Cadence's experiments/session-bench and the research journal.
use crate::feature::types::{Features, Frame, InterpInds, Window};
use crate::ff_config;
use rustfft::{algorithm::Radix4, num_complex::Complex, Fft};

const N: usize = ff_config::MIDI_NUM as usize;

/// |X| of one windowed frame, all SPEC_WINDOW_SIZE bins (the default path's first FFT).
pub fn magnitude_spectrum(window: &Window, window_function: &Window, fft: &Radix4<f32>) -> Vec<f32> {
    let mut buffer = vec![Complex { re: 0.0f32, im: 0.0 }; ff_config::SPEC_WINDOW_SIZE];
    for i in 0..buffer.len() {
        buffer[i].re = window[i] * window_function[i];
    }
    fft.process(&mut buffer);
    buffer.iter().map(|c| (c.re.powf(2.) + c.im.powf(2.)).sqrt()).collect()
}

// Median = the element of rank len/2 in sorted order. Its VALUE does not depend on how it is
// found, so the fast paths below return exactly what a full sort would (no NaN can occur:
// magnitudes are finite and non-negative).
#[inline(always)]
fn cx(v: &mut [f32; 9], a: usize, b: usize) {
    // compare + select rather than f32::min/max: wasm min/max carry NaN and signed-zero
    // rules that cost several instructions each; neither case can occur here.
    let (x, y) = (v[a], v[b]);
    let swap = y < x;
    v[a] = if swap { y } else { x };
    v[b] = if swap { x } else { y };
}

/// Median of exactly 9 values: Paeth's 19-exchange median network.
#[inline(always)]
fn median9(mut v: [f32; 9]) -> f32 {
    cx(&mut v, 1, 2); cx(&mut v, 4, 5); cx(&mut v, 7, 8);
    cx(&mut v, 0, 1); cx(&mut v, 3, 4); cx(&mut v, 6, 7);
    cx(&mut v, 1, 2); cx(&mut v, 4, 5); cx(&mut v, 7, 8);
    cx(&mut v, 0, 3); cx(&mut v, 5, 8); cx(&mut v, 4, 7);
    cx(&mut v, 3, 6); cx(&mut v, 1, 4); cx(&mut v, 2, 5);
    cx(&mut v, 4, 7); cx(&mut v, 4, 2); cx(&mut v, 6, 4);
    cx(&mut v, 4, 2);
    v[4]
}

/// Median of a short slice (`tmp` is scratch space, reused to avoid allocating).
#[inline(always)]
fn median_slice(v: &[f32], tmp: &mut Vec<f32>) -> f32 {
    if v.len() == 9 {
        let mut a = [0f32; 9];
        a.copy_from_slice(v);
        return median9(a);
    }
    tmp.clear();
    tmp.extend_from_slice(v);
    let k = tmp.len() / 2;
    *tmp.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap()).1
}

// Four medians at once where the build has simd128 (the module ffWorker loads on most
// devices). f32x4.pmin(x, y) is defined as (y < x) ? y : x and pmax(x, y) as (x < y) ? y : x:
// exactly the compare-and-select of `cx` above, so the medians, and everything after them,
// are bit-identical to the scalar path. Medians of 9 only (asserted at compile time).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd {
    use core::arch::wasm32::*;
    const _: () = assert!(crate::ff_config::PC_HPSS_TIME_FRAMES == 9 && crate::ff_config::PC_HPSS_FREQ_BINS == 9);

    #[inline(always)]
    fn cx(v: &mut [v128; 9], a: usize, b: usize) {
        let (x, y) = (v[a], v[b]);
        v[a] = f32x4_pmin(x, y);
        v[b] = f32x4_pmax(x, y);
    }

    #[inline(always)]
    fn median9(mut v: [v128; 9]) -> v128 {
        cx(&mut v, 1, 2); cx(&mut v, 4, 5); cx(&mut v, 7, 8);
        cx(&mut v, 0, 1); cx(&mut v, 3, 4); cx(&mut v, 6, 7);
        cx(&mut v, 1, 2); cx(&mut v, 4, 5); cx(&mut v, 7, 8);
        cx(&mut v, 0, 3); cx(&mut v, 5, 8); cx(&mut v, 4, 7);
        cx(&mut v, 3, 6); cx(&mut v, 1, 4); cx(&mut v, 2, 5);
        cx(&mut v, 4, 7); cx(&mut v, 4, 2); cx(&mut v, 6, 4);
        cx(&mut v, 4, 2);
        v[4]
    }

    /// Time medians of frame t (needs frames t-4 ..= t+4) for bins [0, out.len()), out.len()
    /// a multiple of 4.
    pub fn time_row(mags: &[Vec<f32>], t: usize, out: &mut [f32]) {
        let mut k = 0;
        while k + 4 <= out.len() {
            let mut v = [f32x4_splat(0.); 9];
            for j in 0..9 {
                let row = &mags[t + j - 4];
                v[j] = unsafe { v128_load(row.as_ptr().add(k) as *const v128) };
            }
            unsafe { v128_store(out.as_mut_ptr().add(k) as *mut v128, median9(v)) };
            k += 4;
        }
    }

    /// Frequency medians of one spectrum for bins [lo, hi): each needs bins k-4 ..= k+4,
    /// (hi - lo) a multiple of 4.
    pub fn freq_row(mag: &[f32], lo: usize, hi: usize, out: &mut [f32]) {
        let mut k = lo;
        while k + 4 <= hi {
            let mut v = [f32x4_splat(0.); 9];
            for j in 0..9 {
                v[j] = unsafe { v128_load(mag.as_ptr().add(k + j - 4) as *const v128) };
            }
            unsafe { v128_store(out.as_mut_ptr().add(k) as *mut v128, median9(v)) };
            k += 4;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const SIMD: bool = true;
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
const SIMD: bool = false;

/// Median along time of each bin of the non-redundant half spectrum, laid out [t][k]
/// (k = 0..=SPEC_WINDOW_SIZE/2). Each bin's history is copied out once so the sliding
/// window reads contiguous memory instead of PC_HPSS_TIME_FRAMES different frames.
fn time_medians(mags: &[Vec<f32>]) -> Vec<f32> {
    let n = mags.len();
    let hb = ff_config::SPEC_WINDOW_SIZE / 2 + 1;
    let kt = ff_config::PC_HPSS_TIME_FRAMES / 2;
    let mut out = vec![0f32; n * hb];
    // SIMD covers the frames with a full window on both sides and the first 4*floor(hb/4) bins
    let vec_bins = if SIMD && n > 2 * kt { (hb / 4) * 4 } else { 0 };
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if vec_bins > 0 {
        for t in kt..n - kt {
            simd::time_row(mags, t, &mut out[t * hb..t * hb + vec_bins]);
        }
    }
    let mut col = vec![0f32; n];
    let mut tmp: Vec<f32> = Vec::with_capacity(ff_config::PC_HPSS_TIME_FRAMES);
    for k in 0..hb {
        for t in 0..n {
            col[t] = mags[t][k];
        }
        for t in 0..n {
            if k < vec_bins && t >= kt && t + kt < n {
                continue;
            }
            out[t * hb + k] = median_slice(&col[t.saturating_sub(kt)..(t + kt + 1).min(n)], &mut tmp);
        }
    }
    out
}

/// Harmonic part of every frame (soft mask H^2 / (H^2 + P^2), H the median along time, P
/// along frequency, mirrored onto the symmetric half), then the default path's modified
/// autocorrelation of it, interpolated onto the 48 notes, before any octave fix or top-k.
pub(crate) fn harmonic_note_energies(mags: &[Vec<f32>], fft: &Radix4<f32>, interp: &InterpInds) -> Vec<Frame> {
    let mut frames: Vec<Frame> = Vec::with_capacity(mags.len());
    for_each_harmonic_buffer(mags, fft, |buffer| {
        let mut frame: Frame = [0.; N];
        for i in 0..ff_config::SPEC_BINS_NUM as usize {
            let ii = &interp[i];
            let hi = buffer[ii.hi_index].re;
            let lo = buffer[ii.hi_index - 1].re;
            let hi = if hi.signum() == -1. { 0. } else { hi };
            let lo = if lo.signum() == -1. { 0. } else { lo };
            frame[i / ff_config::SPEC_BINS_PER_MIDI as usize] += ii.hi_weight * hi + ii.lo_weight * lo;
        }
        frames.push(frame);
    });
    frames
}

/// The autocorrelation of every frame's harmonic part (the second FFT's output, negative
/// values clamped to 0), handed to `sink` one frame at a time.
fn for_each_harmonic_buffer(mags: &[Vec<f32>], fft: &Radix4<f32>, mut sink: impl FnMut(&[Complex<f32>])) {
    let n = mags.len();
    let w = ff_config::SPEC_WINDOW_SIZE;
    let half = w / 2;
    let hb = half + 1;
    let kf = ff_config::PC_HPSS_FREQ_BINS / 2;
    let h_all = time_medians(mags);
    let mut tmp: Vec<f32> = Vec::with_capacity(ff_config::PC_HPSS_FREQ_BINS);
    let mut buffer = vec![Complex { re: 0.0f32, im: 0.0 }; w];
    let mut p_row = vec![0f32; hb];
    // SIMD block starts k with k - kf >= 0 and k + 3 + kf <= hb - 1
    let (simd_lo, simd_hi) = if SIMD { (kf, kf + ((hb - 2 * kf - 1) / 4) * 4) } else { (0, 0) };
    for t in 0..n {
        let mag = &mags[t];
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        simd::freq_row(mag, simd_lo, simd_hi, &mut p_row);
        for k in 0..hb {
            if k < simd_lo || k >= simd_hi {
                p_row[k] = median_slice(&mag[k.saturating_sub(kf)..(k + kf + 1).min(hb)], &mut tmp);
            }
        }
        for k in 0..hb {
            let h = h_all[t * hb + k];
            let p = p_row[k];
            let mask = if h + p > 0. { h * h / (h * h + p * p) } else { 0. };
            buffer[k] = Complex { re: (mag[k] * mask).cbrt(), im: 0.0 };
            if k > 0 && k < half {
                buffer[w - k] = Complex { re: (mag[w - k] * mask).cbrt(), im: 0.0 };
            }
        }
        fft.process(&mut buffer);
        sink(&buffer);
    }
}

/// Where the autocorrelation buffer is read for a fractional MIDI pitch `x`: the lag above it
/// and the two weights — linear in MIDI between the two lags around it, as
/// compute_interp_inds interpolates. Value = w_hi * ac[hi] + w_lo * ac[hi - 1].
#[derive(Clone, Copy)]
struct Reading {
    hi: usize,
    w_hi: f32,
    w_lo: f32,
}

fn reading(x: f32, sample_rate: u32) -> Reading {
    let half = ff_config::SPEC_WINDOW_SIZE / 2;
    let f = 440. * 2f32.powf((x - 69.) / 12.);
    let lag = sample_rate as f32 / f;
    let hi = (lag.ceil() as usize).clamp(2, half);
    let lo = hi - 1;
    let m = |k: usize| 69. + 12. * ((sample_rate as f32 / k as f32) / 440.).log2();
    let (mlo, mhi) = (m(lo), m(hi));
    let w_hi = (mlo - x) / (mlo - mhi);
    let w_lo = 1. - w_hi;
    Reading { hi, w_hi, w_lo }
}

/// How far the window's tuning sits from A = 440, in semitones (-0.5 .. 0.5): the circular
/// mean of where its energy falls between two equal-tempered notes — every frame's
/// autocorrelation read every tenth of a semitone over MIDI 49.5 - 74.5, each reading
/// weighted by its energy at its angle 2 pi (x - round(x)).
pub fn window_tuning(acs: &[Vec<f32>], sample_rate: u32) -> f32 {
    let (mut cs, mut sn) = (0f32, 0f32);
    let lo = 50.0f32 - 0.5;
    // (reading, cos, sin) of each grid pitch, the same for every frame
    let grid: Vec<(Reading, f32, f32)> = (0..(25 * 10))
        .map(|g| {
            let x = lo + g as f32 * 0.1;
            let ph = 2. * std::f32::consts::PI * (x - x.round());
            (reading(x, sample_rate), ph.cos(), ph.sin())
        })
        .collect();
    for ac in acs {
        for &(r, c, s) in &grid {
            let e = r.w_hi * ac[r.hi] + r.w_lo * ac[r.hi - 1];
            cs += e * c;
            sn += e * s;
        }
    }
    sn.atan2(cs) / (2. * std::f32::consts::PI)
}

/// harmonic_note_energies, retuned: each note's energy read `window_tuning` semitones off
/// the equal-tempered grid, so a session tuned off A = 440 lands on the notes again.
pub(crate) fn tuned_harmonic_note_energies(mags: &[Vec<f32>], fft: &Radix4<f32>, sample_rate: u32) -> Vec<Frame> {
    let mut acs: Vec<Vec<f32>> = Vec::with_capacity(mags.len());
    let half = ff_config::SPEC_WINDOW_SIZE / 2;
    for_each_harmonic_buffer(mags, fft, |buffer| {
        acs.push(buffer[..=half].iter().map(|c| if c.re.signum() == -1. { 0. } else { c.re }).collect());
    });
    let delta = window_tuning(&acs, sample_rate);
    let per = ff_config::SPEC_BINS_PER_MIDI as f32;
    let edge = (per / 2.).floor() / per;
    let bins: Vec<Reading> = (0..ff_config::SPEC_BINS_NUM as usize)
        .map(|b| reading(ff_config::MIDI_LOW as f32 - edge + b as f32 / per + delta, sample_rate))
        .collect();
    acs.iter()
        .map(|ac| {
            let mut fr: Frame = [0.; N];
            for (b, r) in bins.iter().enumerate() {
                fr[b / ff_config::SPEC_BINS_PER_MIDI as usize] += r.w_hi * ac[r.hi] + r.w_lo * ac[r.hi - 1];
            }
            fr
        })
        .collect()
}

fn remove_drone(frames: &mut Vec<Frame>) {
    if frames.is_empty() {
        return;
    }
    let mut col: Vec<f32> = vec![0.; frames.len()];
    for j in 0..N {
        for (f, fr) in frames.iter().enumerate() {
            col[f] = fr[j];
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = col[col.len() / 2];
        for fr in frames.iter_mut() {
            fr[j] = (fr[j] - ff_config::PC_DRONE_FRACTION * med).max(0.);
        }
    }
}

/// Keeps the k strongest of `frame[..len]` (stable order on ties, as the default path).
fn keep_strongest(frame: &mut Frame, len: usize, k: usize) {
    if k >= len {
        return;
    }
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap());
    for i in &order[0..(len - k)] {
        frame[*i] = 0.;
    }
}

/// One window's pitch-class features: frame k holds the energies of the 12 classes in
/// its first 12 slots (C first: MIDI_LOW is a C), the rest zero.
pub fn features_from_magnitudes(mags: &[Vec<f32>], fft: &Radix4<f32>, interp: &InterpInds, tuning: Option<u32>) -> Features {
    let mut frames = match tuning {
        // any pitch (FolkFriend::set_any_pitch): read the notes where this window's tuning puts them
        Some(sample_rate) => tuned_harmonic_note_energies(mags, fft, sample_rate),
        None => harmonic_note_energies(mags, fft, interp),
    };
    remove_drone(&mut frames);
    frames
        .iter()
        .map(|fr| {
            let mut pc: Frame = [0.; N];
            for j in 0..N {
                pc[j % 12] += fr[j];
            }
            keep_strongest(&mut pc, 12, ff_config::PC_RETAINED_CLASSES);
            pc
        })
        .collect()
}

/// A contour folded onto the 12 pitch classes: note n becomes the n-th symbol modulo 12,
/// so every C is 'a', every C# 'b', … every B 'l'.
pub fn fold_contour_string(contour: &str) -> String {
    contour
        .chars()
        .filter_map(|c| ff_config::CONTOUR_TO_QUERY_CHAR.iter().position(|&q| q == c))
        .map(|i| ff_config::CONTOUR_TO_QUERY_CHAR[i % 12])
        .collect()
}
