//! First-principles 4-organ physical AI Vocal Box (`vocal_box`).
//!
//! Models the complete human vocal apparatus using only its four irreducible physical elements:
//! 1. **Subglottal Supply** (`lungs` + `trachea`),
//! 2. **Myoelastic-Aerodynamic Glottal Valve** ([`GlottalValve`]: progressive-zipper [`RibbonGlottis`]
//!    or coupled 2-DOF self-oscillating [`TwoMassGlottis`] across `glottis_port`),
//! 3. **Articulated Supraglottal Waveguide** (`pharynx` + `oral` soft-wall tubes, 2D midsagittal
//!    contact mechanics, Reynolds constriction & incisor turbulence),
//! 4. **Velopharyngeal Bifurcation** (`velum_port` $\to$ `nasal` + sinus $\to$ nostrils).

pub mod glottal;
pub mod voice;

pub use crate::anatomy::human::{BodyMix, BodyName};
pub use crate::sources::two_mass::{TwoMassGlottis, zipper_area_c1};
pub use glottal::{
    BodyAnatomy, GlottalValve, Registration, RibbonGlottis, register_state, ribbon_pulse_c1,
};
pub use voice::{EmotionPreset, EmotionProfile, VocalBoxTelemetry, VocalBoxVoice};
