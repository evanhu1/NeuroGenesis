use std::f32::consts::PI;
use types::ActivationFunction;

/// Apply a heritable transfer function to a hidden neuron's membrane state.
///
/// Every output lies in [-1, 1] — an invariant of the recurrent substrate (see
/// [`ActivationFunction`]), not an assumption inherited from tanh. Six
/// functions are naturally bounded; the `Saturating*` variants are the
/// saturating forms of the unbounded functions in the original
/// weight-agnostic set, which was strictly feedforward and never faced
/// recurrent gain.
#[inline(always)]
pub fn apply(function: ActivationFunction, x: f32) -> f32 {
    match function {
        ActivationFunction::Tanh => crate::fast_tanh(x),
        ActivationFunction::SaturatingLinear => x.clamp(-1.0, 1.0),
        ActivationFunction::Step => f32::from(x > 0.0),
        ActivationFunction::Sin => (PI * x).sin(),
        ActivationFunction::Cos => (PI * x).cos(),
        ActivationFunction::Gaussian => (-x * x / 2.0).exp(),
        ActivationFunction::Sigmoid => 0.5 * (crate::fast_tanh(0.5 * x) + 1.0),
        ActivationFunction::SaturatingInverse => (-x).clamp(-1.0, 1.0),
        ActivationFunction::SaturatingAbs => x.abs().min(1.0),
        ActivationFunction::SaturatingRelu => x.clamp(0.0, 1.0),
    }
}

/// Local derivative of [`apply`] with respect to the membrane state `x`, given
/// `a == apply(function, x)`. This is the postsynaptic gain used by
/// eligibility traces. Saturating functions have zero gain outside their
/// linear region; step uses the boxcar pseudo-derivative `max(0, 1 - |x|)`
/// (the e-prop convention for non-differentiable neurons).
#[inline(always)]
pub fn derivative(function: ActivationFunction, x: f32, a: f32) -> f32 {
    match function {
        ActivationFunction::Tanh => 1.0 - a * a,
        ActivationFunction::SaturatingLinear => f32::from(x.abs() < 1.0),
        ActivationFunction::Step => (1.0 - x.abs()).max(0.0),
        ActivationFunction::Sin => PI * (PI * x).cos(),
        ActivationFunction::Cos => -PI * (PI * x).sin(),
        ActivationFunction::Gaussian => -x * a,
        ActivationFunction::Sigmoid => a * (1.0 - a),
        ActivationFunction::SaturatingInverse => -f32::from(x.abs() < 1.0),
        ActivationFunction::SaturatingAbs => {
            if x.abs() < 1.0 {
                x.signum()
            } else {
                0.0
            }
        }
        ActivationFunction::SaturatingRelu => f32::from(x > 0.0 && x < 1.0),
    }
}
