//! First-principles 4-organ physical AI Vocal Box (`vocal_box`).
//!
//! Models the complete human vocal apparatus using only its four irreducible physical elements:
//! 1. **Subglottal Supply** (`lungs` + `trachea`),
//! 2. **Coupled Vocal Folds** ([`TwoMassGlottis`] self-oscillating across `glottis_port`),
//! 3. **Articulated Supraglottal Waveguide** (`pharynx` + `oral` soft-wall tubes, 2D midsagittal
//!    contact mechanics, Reynolds constriction & incisor turbulence),
//! 4. **Velopharyngeal Bifurcation** (`velum_port` $\to$ `nasal` + sinus $\to$ nostrils).

pub mod glottal;
pub mod voice;

pub use crate::anatomy::human::{BodyMix, BodyName};
pub use crate::sources::two_mass::{TwoMassGlottis, zipper_area_c1};
pub use glottal::{BodyAnatomy, Registration, register_state};
pub use voice::{EmotionPreset, EmotionProfile, VocalBoxTelemetry, VocalBoxVoice};
