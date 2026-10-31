use crate::feature::types::{Frame, InterpInds, Window};
use crate::ff_config;
use rustfft::{Fft, algorithm::Radix4, num_complex::Complex};

pub fn modified_autocorrelation(
    window: Window,
    window_function: &Window, 
    fft: &Radix4<f32>,
    interp_inds: &InterpInds,
) -> Frame {
    // This is the most important signal processing step of folkfriend.
    //  starting with a 1024 sample long slice of audio signal data, use
    //  Fourier transforms to compute the 'enhanced' autocorrelation, and
    //  then perform some denoising / filtering steps to create a vector
    //  of MIDI_NUM features, where each feature is the energy associated
    //  with that MIDI pitch (i.e., musical note).

    // Define empty buffer for use with this frame
    let mut buffer: [Complex<f32>; ff_config::SPEC_WINDOW_SIZE] =
        [Complex { re: 0.0, im: 0.0 }; ff_config::SPEC_WINDOW_SIZE];
    // =============================
    // === Apply window function ===
    // =============================

    for i in 0..buffer.len() {
        buffer[i].re = window[i] * window_function[i];
    }

    // ===============================
    // === Compute autocorrelation ===
    // ===============================

    // Fourier transform of windowed signal, X(w)
    fft.process(&mut buffer);

    // First, compute power spectrum, which is exactly X(w)X*(w), which is
    //  the same as |X|^2. Then, take the sextic root of the power spectrum
    //  (which is the same as directly computing cube root |X|). This step
    //  corresponds to a "k-value" of 1/3, see
    //  https://labrosa.ee.columbia.edu/~dpwe/papers/ToloK2000-mupitch.pdf
    //  (which recommends k as 2/3) for this magnitude compression.
    for i in 0..buffer.len() {
        // Upstream's note here says this line is "~1.75 faster than using
        // norm()", with `norm().cbrt()` left commented out above it. That was
        // measured on a native target. On wasm32 the trade-off inverts:
        // `f32.sqrt` is a single machine instruction while `powf` is a libm
        // call, and this runs 1024 times per PCM frame — 720k times for one 15s
        // analysis window. Writing the sixth root as sqrt-then-cbrt (the same
        // value: s^(1/6) = cbrt(sqrt(s))) takes the whole feed stage from
        // 35.9ms to 14.7ms per window, i.e. -10% on a full window analysis.
        //
        // NOT bit-identical — float rounding differs, and 79 windows in 80 see
        // their transcribed contour change by a character or two. Validated the
        // way this project validates every detection change: full backtest over
        // the 4 annotated sessions, 3491 windows — recall 97/118 unchanged,
        // 16 false positives unchanged, 114 results unchanged, not one tune
        // gained or lost. (`powf(x, 2.0)` is left alone: the compiler already
        // lowers it to a multiplication, measured, so rewriting it gains
        // nothing and only widens the diff.)
        buffer[i].re = (buffer[i].re.powf(2.) + buffer[i].im.powf(2.)).sqrt().cbrt();
        buffer[i].im = 0.;
    }

    // Fourier transform of magnitude-compressed power spectrum.
    //  Note! Fourier transform is basically the same as inverse
    //  fourier transform (we have real and even functions here).
    //  We can reuse the exact same FFT object.
    //  See Tom's old 3G3 jotter for some calculations on this.
    fft.process(&mut buffer);

    // =============================================
    // === Compute features from autocorrelation ===
    // =============================================

    // Peak pruning. At this point buffer[i].im = 0 for all i.
    for i in 0..buffer.len() {
        // TODO is the compiler smart enough that the if branch isn't
        //  inefficient here? Or is max(0, x) much better?
        if buffer[i].re.signum() == -1. {
            buffer[i].re = 0.;
        }
    }

    // We have now spectral energy at the frequencies described above.
    //  Use linear interpolation to find the energy at the frequencies of
    //  each of the MIDI notes in the range of MIDI values that folkfriend
    //  uses.
    let mut frame: Frame = [0.; ff_config::MIDI_NUM as usize];
    for i in 0..ff_config::SPEC_BINS_NUM as usize {
        // Perform linear interpolation
        let hi_wt = interp_inds[i].hi_weight;
        let lo_wt = interp_inds[i].lo_weight;
        let hi_ac = buffer[interp_inds[i].hi_index].re;
        let lo_ac = buffer[interp_inds[i].hi_index - 1].re;
        let feature = hi_wt * hi_ac + lo_wt * lo_ac;
        frame[i / ff_config::SPEC_BINS_PER_MIDI as usize] += feature;
    }

    // ==========================================
    // === Postfiltering of computed features ===
    // ==========================================

    // Octave fixing

    // This next little algorithm is a bit funny. It's not as complicated
    //  as it sounds. Any note 12 MIDI notes (1 octave) higher than another
    //  is likely to represent a timbre containing harmonics. We don't want
    //  to individually track each harmonic, rather extract the single
    //  fundemental frequency. In this algorithm, if a note N1 is an octave
    //  lower than a note N2, and E1 < E2, where E is the energy of a note,
    //  then "shift" the lower harmonic up an octave by setting
    //  E2 += E1, E1 = 0. If E1 > E2, do nothing.
    //  If we have the same situation spanning two octaves, with notes N1,
    //  N2, N3, and E1 < E2 < E3 then shift the energy up to E3, i.e. set
    //  E3 = E2 + E1, E1 = 0, E2 = 0. However, if E2 < E1 < E3 then shift
    //  E2 to E3 but leave E1 unchanged. This generalises naturally to an
    //  arbitrary number of octaves.

    //  This algorithm preserves the total energy of the spectrogram, and
    //  only ever shifts energy around by moving it by whole octaves.

    // Low indices = low frequencies.
    // Fixed-size buffers rather than two fresh Vecs per musical key. This runs
    // once per PCM frame, i.e. 703 times for a 15s analysis window, so the old
    // form heap-allocated ~17k times per window for at most 4 entries each.
    // The two-phase structure is load-bearing and preserved: every comparison
    // is taken against the ORIGINAL frame, before any energy is shifted.
    const MAX_OCTAVES: usize = (ff_config::MIDI_NUM as usize) / 12 + 1;
    for musical_key in 0..12 {
        let mut inds: [usize; MAX_OCTAVES] = [0; MAX_OCTAVES];
        let mut should_shift: [bool; MAX_OCTAVES] = [false; MAX_OCTAVES];
        let mut count = 0;

        let mut ind = musical_key;
        while ind < frame.len() - 12 {
            should_shift[count] = frame[ind + 12] > frame[ind];
            inds[count] = ind;
            count += 1;
            ind += 12;
        }

        for i in 0..count {
            // basically if true, move up an octave and set to zero
            // if not true, continue
            if should_shift[i] {
                frame[inds[i] + 12] += frame[inds[i]];
                frame[inds[i]] = 0.;
            }
        }
    }

    // Noise filtering

    // Simple algorithm here - retain only the 5 most energetic features
    //  each frame. This significantly speeds up the later decoder step.

    // Tom is not very good at Rust. There's probably some things to be
    //  improved here.

    // Sort an index array in place instead of collecting two Vecs per frame
    // (again: 703 frames per analysis window). `sort_by` is stable in both
    // forms, so equal energies keep ascending index order exactly as before,
    // and the same 43 features get zeroed.
    let mut order: [usize; ff_config::MIDI_NUM as usize] = [0; ff_config::MIDI_NUM as usize];
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    order.sort_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap());

    for i in &order
        [0..(ff_config::MIDI_NUM as usize - ff_config::RETAINED_FEATURES_PER_FRAME as usize)]
    {
        frame[*i] = 0.;
    }

    return frame;
}
