pub mod abc;
pub mod decode;
pub mod feature;
pub mod ff_config;
pub mod index;
pub mod query;

use console_error_panic_hook;
use js_sys;
use serde_json::json;
use std::convert::TryInto;
use std::panic;
use std::slice;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

pub struct FolkFriend {
    query_engine: query::QueryEngine,
    feature_extractor: feature::FeatureExtractor,
    feature_decoder: decode::FeatureDecoder,
    pitch_class_mode: bool,
}

impl FolkFriend {
    pub fn new() -> FolkFriend {
        FolkFriend {
            query_engine: query::QueryEngine::new(),
            feature_extractor: feature::FeatureExtractor::new(ff_config::SAMPLE_RATE_DEFAULT)
                .unwrap(),
            feature_decoder: decode::FeatureDecoder::new(ff_config::SAMPLE_RATE_DEFAULT).unwrap(),
            pitch_class_mode: false,
        }
    }

    pub fn version(&self) -> String {
        ff_config::VERSION.to_string()
    }

    pub fn load_index_from_json_string(&mut self, json_string: String) {
        let tune_index = index::tune_index_from_string(&json_string);
        self.query_engine.use_tune_index(tune_index);
    }

    pub fn set_sample_rate(
        &mut self,
        sample_rate: u32,
    ) -> Result<(), feature::signal::SampleRateError> {
        if self.feature_extractor.sample_rate != sample_rate {
            self.feature_extractor = feature::FeatureExtractor::new(sample_rate)?;
            self.feature_extractor.pitch_class_mode = self.pitch_class_mode;
        }
        if self.feature_decoder.sample_rate != sample_rate {
            self.feature_decoder = decode::FeatureDecoder::new(sample_rate)?;
        }
        Ok(())
    }

    pub fn feed_entire_pcm_signal(&mut self, pcm_signal: Vec<f32>) {
        self.feature_extractor.feed_signal(pcm_signal);
    }

    pub fn feed_single_pcm_window(&mut self, pcm_window: [f32; ff_config::SPEC_WINDOW_SIZE]) {
        self.feature_extractor.feed_window(pcm_window);
    }

    pub fn flush_pcm_buffer(&mut self) {
        self.feature_extractor.flush();
    }

    /// Switches this instance to pitch-class recognition (feature/pitch_class.rs):
    /// transcriptions become pitch-class contours and the index is folded to
    /// match, now or when loaded. One-way, per instance: an instance that also
    /// serves take_features (score following, tempo) must stay in the default mode.
    pub fn set_pitch_class_mode(&mut self) {
        self.pitch_class_mode = true;
        self.feature_extractor.pitch_class_mode = true;
        self.feature_extractor.flush();
        self.query_engine.use_pitch_classes();
    }

    /// Pitch-class mode (Cadence, 2026-11-07): "any pitch" — each window's tuning estimated and
    /// corrected (a session tuned up to a quarter-tone off A = 440 lands on the notes again) and
    /// the tune looked for in all twelve keys (one played in another key than the index's).
    /// Measured: off the written pitch (a quarter-tone up) 91/96 tunes instead of 8/96; at the
    /// written pitch the same tunes and false positives, with a few more wrong names shown for
    /// a moment; about 1.5x the time of a window. Off by default.
    pub fn set_any_pitch(&mut self, on: bool) {
        self.feature_extractor.pitch_class_tuning = on;
        self.query_engine.set_any_key(on);
    }

    fn transcribe_pitch_classes(
        &mut self,
    ) -> Result<(decode::types::ContourString, decode::ContourDebugFeatures), decode::DecoderError> {
        let (mut features, tuning) = self.feature_extractor.pitch_class_features();
        self.feature_extractor.flush();
        self.feature_decoder
            .decode_pitch_class_contour(&mut features)
            .map(|(contour, mut debug)| {
                debug.tuning = tuning;
                (contour, debug)
            })
    }

    pub fn transcribe_pcm_buffer(
        &mut self,
    ) -> Result<decode::types::ContourString, decode::DecoderError> {
        if self.pitch_class_mode {
            return self.transcribe_pitch_classes().map(|(contour, _)| contour);
        }
        let lattice_path = self
            .feature_decoder
            .decode_lattice_path(&mut self.feature_extractor.features)?;
        let contour = self
            .feature_decoder
            .decode_contour(&lattice_path, &self.feature_extractor.features);
        self.feature_extractor.flush();
        return contour;
    }

    // Debug/experimental instrumentation (2026-08-18) — never called from the
    // production `transcribe_pcm_buffer` path, purely additive. See
    // `FolkFriendWASM::transcribe_pcm_buffer_debug`.
    pub fn transcribe_pcm_buffer_debug(
        &mut self,
    ) -> Result<(decode::types::ContourString, decode::ContourDebugFeatures), decode::DecoderError>
    {
        if self.pitch_class_mode {
            return self.transcribe_pitch_classes();
        }
        let lattice_path = self
            .feature_decoder
            .decode_lattice_path(&mut self.feature_extractor.features)?;
        let result = self
            .feature_decoder
            .decode_contour_debug(&lattice_path, &self.feature_extractor.features);
        self.feature_extractor.flush();
        return result;
    }

    pub fn run_transcription_query(
        &self,
        contour: &decode::types::ContourString,
    ) -> Result<query::TranscriptionQueryResults, query::QueryError> {
        self.query_engine.run_contour_query(contour)
    }

    pub fn run_name_query(
        &self,
        query: &String,
    ) -> Result<query::NameQueryResults, query::QueryError> {
        self.query_engine.run_name_query(query)
    }

    pub fn contour_to_abc(&self, contour_string: &decode::types::ContourString) -> String {
        let contour: decode::types::Contour =
            decode::types::contour_string_to_contour(contour_string);
        return abc::contour_to_abc(&contour);
    }

    pub fn settings_from_tune_id(
        &self,
        tune_id: index::schema::TuneID,
    ) -> Result<Vec<(index::schema::SettingID, index::schema::Setting)>, query::QueryError> {
        self.query_engine.settings_from_tune_id(tune_id)
    }

    pub fn aliases_from_tune_id(
        &self,
        tune_id: index::schema::TuneID,
    ) -> Result<Vec<String>, query::QueryError> {
        self.query_engine.aliases_from_tune_id(tune_id)
    }
}

// Define a WASM-safe version of this FolkFriend class. The key point is that
//  all function signatures will have simple types that can pass between WASM
//  and JS. e.g. search results are encoded as a JSON String rather than a
//  vector of structs. The latter would be better, but WASM/JS can't hadnle
//  this yet.

#[wasm_bindgen]
pub struct FolkFriendWASM {
    ff: FolkFriend,
}

#[wasm_bindgen]
impl FolkFriendWASM {
    #[wasm_bindgen(constructor)]
    pub fn new() -> FolkFriendWASM {
        panic::set_hook(Box::new(console_error_panic_hook::hook));
        FolkFriendWASM {
            ff: FolkFriend::new(),
        }
    }

    pub fn version(&self) -> String {
        self.ff.version()
    }

    pub fn load_index_from_json_obj(&mut self, js_value: JsValue) {
        // This function doesn't call the underlying self.ff.load_index_from_string.
        let tune_index: index::TuneIndex = serde_wasm_bindgen::from_value(js_value).unwrap();
        self.ff.query_engine.use_tune_index(tune_index);
    }

    // ── Detector parameters ─────────────────────────────────────────────────
    // The engine is a generic detector: every default below is upstream
    // FolkFriend's behaviour, and the CALLER sets the values it has tuned
    // (Cadence: src/session/sessionConfig.ts). With no call, results are
    // bit-identical to the unparameterised engine.

    /// Tempo search range, crotchets per minute, upper bound exclusive, step 5.
    /// Default 60 / 240.
    pub fn set_tempo_range(&mut self, low_bpm: u32, high_bpm: u32) {
        decode::set_tempo_range(low_bpm, high_bpm);
    }

    /// Notes weaker than `min_power` or shorter than `min_duration_frames`
    /// frames are dropped before quantisation. Default 0.10 / 3.
    pub fn set_note_filter(&mut self, min_power: f32, min_duration_frames: u32) {
        decode::set_note_filter(min_power, min_duration_frames);
    }

    /// Tempo selection model: per-note score `intercept - slope * quavers`.
    /// Default 3.0 / 0.5.
    pub fn set_tempo_model(&mut self, intercept: f32, slope: f32) {
        decode::set_tempo_model(intercept, slope);
    }

    /// Settings re-aligned by Needleman-Wunsch after the n-gram pass.
    /// Default ff_config::QUERY_REPASS_SIZE (2000).
    pub fn set_num_repass(&mut self, num_repass: usize) {
        self.ff.query_engine.set_num_repass(num_repass);
    }

    /// Queries shorter than this many contour symbols return no candidates.
    /// Default 0.
    pub fn set_min_query_length(&mut self, min_query_length: usize) {
        self.ff.query_engine.set_min_query_length(min_query_length);
    }

    /// Pitch-class recognition (Cadence, 2026-11-07), one-way: see
    /// FolkFriend::set_pitch_class_mode. Contours this instance returns are then
    /// pitch classes ('a' = C ... 'l' = B) and its index is folded to match.
    pub fn set_pitch_class_mode(&mut self) {
        self.ff.set_pitch_class_mode();
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) -> bool {
        match self.ff.set_sample_rate(sample_rate) {
            Ok(()) => true,
            Err(_) => false,
        }
    }

    pub fn feed_entire_pcm_signal(&mut self) {
        panic!("Cannot feed entire PCM signal from WebAssembly!");
    }

    pub fn alloc_single_pcm_window(&mut self) -> *mut f32 {
        // This function is not required in the non-WASM version of FolkFriend.
        //  This is required for passing Float32Arrays (audio data) from
        //   javascript into rust.
        let mut buf: [f32; ff_config::SPEC_WINDOW_SIZE] = [0.; ff_config::SPEC_WINDOW_SIZE];

        // This pointer will be given to JavaScript so that it can write
        //  directly into WebAssembly's memory
        let ptr = buf.as_mut_ptr();
        // Take ownership of this memory block to prevent destruction
        // See https://radu-matei.com/blog/practical-guide-to-wasm-memory/
        std::mem::forget(buf);

        return ptr;
    }

    pub fn get_allocated_pcm_window(&mut self, ptr: *mut f32) -> js_sys::Float32Array {
        return unsafe {
            js_sys::Float32Array::view(slice::from_raw_parts(ptr, ff_config::SPEC_WINDOW_SIZE))
        };
    }

    pub fn feed_single_pcm_window(&mut self, ptr: *mut f32) {
        let pcm_window = unsafe { slice::from_raw_parts(ptr, ff_config::SPEC_WINDOW_SIZE) };
        self.ff
            .feed_single_pcm_window(pcm_window.try_into().unwrap());
    }

    pub fn flush_pcm_buffer(&mut self) {
        self.ff.flush_pcm_buffer();
    }

    pub fn transcribe_pcm_buffer(&mut self) -> String {
        match self.ff.transcribe_pcm_buffer() {
            Ok(transcription) => transcription,
            Err(_) => json!({
                "error": "Could not detect any notes".to_string()
            })
            .to_string(),
        }
    }

    // Debug/experimental instrumentation (2026-08-18) — `transcribe_pcm_buffer`
    // above is completely untouched; this is a new, separate export. Returns
    // `{"contour": ..., "features": {...}}` on success (features = note/tempo
    // stats otherwise discarded), or `{"error": ..., "features": null}` on
    // the same "too few notes" case `transcribe_pcm_buffer` reports as a bare
    // error — `features: null` here (rather than omitting note/tempo stats
    // entirely) is deliberate: a noise-study harness needs to know this
    // window produced literally zero usable notes, not just skip the row.
    pub fn transcribe_pcm_buffer_debug(&mut self) -> String {
        match self.ff.transcribe_pcm_buffer_debug() {
            Ok((transcription, features)) => json!({
                "contour": transcription,
                "features": features,
            })
            .to_string(),
            Err(_) => json!({
                "error": "Could not detect any notes".to_string(),
                "features": serde_json::Value::Null,
            })
            .to_string(),
        }
    }

    // Score following (Cadence, 2026-10-28) — additive, like the `_debug`
    // exports: hands the per-frame pitch energies fed so far to JS, row-major
    // (frames × MIDI_NUM, MIDI_LOW first), undecoded and unnormalised, then
    // flushes them. Lets the caller align a whole performance against a known
    // score instead of transcribing it. Production paths untouched.
    pub fn take_features(&mut self) -> js_sys::Float32Array {
        let flat: Vec<f32> = self
            .ff
            .feature_extractor
            .features
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .collect();
        self.ff.feature_extractor.flush();
        js_sys::Float32Array::from(&flat[..])
    }

    /// Score following (Cadence, 2026-11-07, feature/clean.rs): from now on this instance
    /// streams the frames it is fed into cleaned 48-note energies — the harmonic part, as the
    /// pitch-class engine reads it — instead of take_features'. Measured on Cadence's
    /// annotated sessions (experiments/score-follow): with take_clean_features(10 s of
    /// frames, false, 1 s, 3), 17 tunes followed over less than 80 % of their span instead of
    /// 20, the ends nearer the annotations, the passes more regular, and a wrong score
    /// claiming 12 % of a span at the 90th percentile instead of 89 % — in a long recording,
    /// half as many tunes found that were never played.
    pub fn enable_clean_features(&mut self) {
        self.ff.feature_extractor.flush();
        self.ff.feature_extractor.clean_mode = true;
    }

    /// The cleaned frames fed since the last call (frames x 48, as take_features): each note
    /// minus its median over `drone_frames` frames (centred, or the frames up to it when
    /// `causal`; 0 = nothing removed), the median taken every `stride` frames, then the
    /// `top_k` strongest notes of each frame kept (0 = all). Flushes.
    pub fn take_clean_features(&mut self, drone_frames: usize, causal: bool, stride: usize, top_k: usize) -> js_sys::Float32Array {
        let fe = &mut self.ff.feature_extractor;
        let mut frames = fe.clean.finish(&fe.fft, &fe.interp_inds);
        feature::clean::remove_drone_sliding(&mut frames, drone_frames, causal, stride);
        for f in frames.iter_mut() {
            feature::clean::keep_strongest(f, top_k);
        }
        let flat: Vec<f32> = frames.iter().flat_map(|f| f.iter().copied()).collect();
        js_sys::Float32Array::from(&flat[..])
    }

    /// Pitch-class mode: recognise a tune whatever the key and the tuning it is played in (on),
    /// or at the index's own pitch (off, the default). See FolkFriend::set_any_pitch.
    pub fn set_any_pitch(&mut self, on: bool) {
        self.ff.set_any_pitch(on);
    }

    pub fn run_transcription_query(&self, contour_string: &str) -> String {
        // TODO proper error propagation in JSON back to javascript
        let result = self.ff.run_transcription_query(&contour_string.to_string());
        if let Ok(mut query_result) = result {
            // Pass back fewer results to the App than the backend is
            //  configured to provide. Users don't have time to scroll
            //  through a hundred results, even if we do want this sort
            //  of granularity when assessing dataset performance.
            query_result.truncate(20);
            return json!(query_result).to_string();
        } else {
            return json!({
                "error": "Transcription query error".to_string()
            })
            .to_string();
        }
    }

    // Debug/experimental instrumentation (2026-08-18) — identical query to
    // `run_transcription_query` above (same `QueryEngine::run_contour_query`
    // call, already computes/sorts up to `num_output`=100 scored settings
    // internally), just without the `.truncate(20)` applied for the
    // production UI. `run_transcription_query` is completely untouched.
    pub fn run_transcription_query_debug(&self, contour_string: &str) -> String {
        let result = self.ff.run_transcription_query(&contour_string.to_string());
        if let Ok(query_result) = result {
            return json!(query_result).to_string();
        } else {
            return json!({
                "error": "Transcription query error".to_string()
            })
            .to_string();
        }
    }

    pub fn run_name_query(&self, query: &str) -> String {
        // TODO proper error propagation in JSON back to javascriptx
        let result = self.ff.run_name_query(&query.to_string());
        if let Ok(query_result) = result {
            return json!(query_result).to_string();
        } else {
            json!({
                "error": "Name query error".to_string()
            })
            .to_string()
        }
    }
    pub fn contour_to_abc(&self, contour_string: &str) -> String {
        self.ff.contour_to_abc(&contour_string.to_string())
    }

    pub fn settings_from_tune_id(&self, tune_id: index::schema::TuneID) -> String {
        // TODO proper error propagation in JSON back to javascript
        let result = self.ff.settings_from_tune_id(tune_id);
        if let Ok(settings) = result {
            return json!(settings).to_string();
        } else {
            json!({
                "error": "Setting ID query error".to_string()
            })
            .to_string()
        }
    }

    pub fn aliases_from_tune_id(&self, tune_id: index::schema::TuneID) -> String {
        // TODO proper error propagation in JSON back to javascript
        let result = self.ff.aliases_from_tune_id(tune_id);
        if let Ok(aliases) = result {
            return json!(aliases).to_string();
        } else {
            json!({
                "error": "Aliases from tune ID query error".to_string()
            })
            .to_string()
        }
    }
}
