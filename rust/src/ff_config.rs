// Stores important global parameters that are re-used across FolkFriend.
// MAKE SURE THIS MATCHES CARGO.TOML
pub const VERSION: &str = "1.3.0";

// ====================================
// === Signal Processing Paramaters ===
// ====================================

// The range of MIDI pitches that folkfriend supports. These extrema are 
//  inclusive, i.e. the support is [LOW_MIDI, HIGH_MIDI].
pub const MIDI_HIGH: u32 = 95;                      // B6 (1975.5 Hz), just over two octaves above violin open A
pub const MIDI_LOW: u32 = 48;                       // C2 (130.81 Hz), an octave below middle C
pub const MIDI_NUM: u32 = MIDI_HIGH - MIDI_LOW + 1; // =48

// The features are formed by computing the values at SPEC_BINS_PER_MIDI
//  different frequencies, linearly spaced from the frequency of MIDI note
//  MIDI_LOW - edge to frequency MIDI_HIGH + edge
pub const SPEC_BINS_PER_MIDI: u32 = 3;
pub const SPEC_BINS_NUM: u32 = SPEC_BINS_PER_MIDI * MIDI_NUM;

// Here, "edge" is given by ..., e.g.
// SPEC_BINS_PER_MIDI   Lowest MIDI value   Edge
//  1                   48.0                0.0
//  2                   47.75               0.25
//  3                   47.66               0.33
//  4                   47.


// Samples sizes for  short-time Fourier transform (STFT) window width. Note
//  that 48000 / 1024 = 46.875 fps, but the framerate may vary slightly with
//  sample rate. Folkfriend doesn't presume tempo so this isn't problematic.
pub const SPEC_WINDOW_SIZE: usize = 1024;

// The input sample rate is not fixed. In practice, it will almost always be
//  48 kHz or 44.1 kHz. The input signal is not resampled before processing
//  but the sample rate is taken into account during frequency analysis.
//  However, the sample rate must be sufficient to capture the range of pitches
//  specified above.

// The highest frequency that can be encoded by a signal at sample rate N is
//  the frequency N/2 ("Nyquist frequency"). To capture note B6, 1975.53 Hz, a
//  sample rate of *at least* 3952 Hz is required.
pub const SAMPLE_RATE_MIN: u32 = 3952;

// The frequency domain analysis is a modified autocorrelation. Bin N of the
//  autocorrelation corresponds to the frequency sample_rate / N. The spectral
//  window size is fixed however. The maximum bin is the 512th bin so the
//  lowest frequency is sample_rate / 512. To capture C2, 130.81 Hz the sample
//  rate can be no higher than 66,974 Hz.
pub const SAMPLE_RATE_MAX: u32 = 66_974;

// Initialise feature extraction with this. It's usually this.
pub const SAMPLE_RATE_DEFAULT: u32 = 48_000;

// Retain only this many features
pub const RETAINED_FEATURES_PER_FRAME: u32 = 5;
pub const PITCH_MODEL_WEIGHT: f32 = 0.050;
pub const PITCH_MODEL_SHIFT: f32 = -7.0;
pub const BASE_ENERGY_SCORE: f32 = -0.18;

// Why do we cap this? Because otherwise score can decay very low after long silence
//  and then miss the first couple of notes when music actually starts.
// pub const MIN_LATTICE_SCORE: f32 = -5.0;

pub const MIN_NOTE_DURATION: usize = 3;
pub const MIN_NOTE_DURATION_REL: f32 = 0.2;
pub const MIN_NOTE_POWER: f32 = 0.10;


pub const CONTOUR_TO_QUERY_CHAR: [char; MIDI_NUM as usize] = [
    'a', 'b', 'c', 'd', 'e', 
    'f', 'g', 'h', 'i', 'j', 
    'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 
    'u', 'v', 'w', 'x', 'y', 
    'z', 'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V'
];

// Just found in testing that with musical notes 4 is better
//  and with searching strings for names 3 seems to be better :)
pub const QUERY_NGRAM_SIZE_CONTOUR: usize = 4;  // "quadgram"
pub const QUERY_NGRAM_SIZE_NAME: usize = 3;     // "trigram"
// KEPT AT 2000 (Cadence, 2026-09-01). Lowering it is the single biggest
// speed lever — pass 2 is ~80% of a query and is linear in this number — and
// 500 was measured safe on the criterion that looked right: all 57 windows
// with a confident top-1 kept their winner. It is NOT safe. Full backtest on
// the 4 annotated sessions: recall 97/118 at 2000, 94/118 at 1000, 93/118 at
// 500 — real detections lost, reproducibly (the old engine re-run scores the
// same 97, so this is not its tie-order noise).
//
// The mechanism is not shortlist membership, which is what one naturally
// checks: run_contour_query deduplicates by tune_id and keeps each tune's
// BEST-scoring setting. A popular tune has many settings; cutting the
// shortlist cuts how many of them get aligned, so the maximum over them —
// the score the tune is finally reported with — drops. A tune sitting just
// above the detector threshold falls under it. Ranking the shortlist better
// does not help (tried: match-density normalisation put the true tune at
// rank <=243 every time, and still lost the same detections).
pub const QUERY_REPASS_SIZE: usize = 2000;

// ======================================================
// === Pitch-class recognition (Cadence, 2026-11-07) ===
// ======================================================
// Used only once a FolkFriend instance is switched to pitch classes
// (FolkFriend::set_pitch_class_mode); see feature/pitch_class.rs. Chosen on Cadence's
// development sessions, each in the middle of a plateau, then confirmed on sessions held
// out of that choice.

// Harmonic/percussive separation: median filter lengths, frames (9 x 21 ms = 190 ms,
// shorter than a reel's quaver would erase quick notes) and FFT bins (9 x 47 Hz).
pub const PC_HPSS_TIME_FRAMES: usize = 9;
pub const PC_HPSS_FREQ_BINS: usize = 9;
// Fraction of each note's median energy over the window removed as drone / background.
pub const PC_DRONE_FRACTION: f32 = 1.0;
// Pitch classes kept per frame (3 loses the melody under accompaniment, 12 drowns it).
pub const PC_RETAINED_CLASSES: usize = 5;
/// Pitch-class query in every key (2026-11-07): besides the QUERY_REPASS_SIZE settings the
/// written key's n-gram pass keeps, the best this many (setting, transposition) pairs of the
/// other eleven keys are aligned too (query/heuristic.rs, run_transcription_query_any_key).
/// 1000 measured as good as 2000 (in tune: 675/698 FP 14 vs 674 FP 13; a quarter-tone sharp:
/// 92/96 vs 91/96), 500 not (a quarter-tone sharp: twice the wrong names shown for a moment).
pub const PC_ANY_KEY_EXTRA: usize = 1000;
