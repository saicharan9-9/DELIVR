/**
 * DELIVR - AI Delivery Time Prediction
 * Client-side micro-interactions & animations
 */

document.addEventListener('DOMContentLoaded', () => {
  // Smooth scroll and focus enhancements
  const numberInputs = document.querySelectorAll('input[type="number"]');
  numberInputs.forEach(input => {
    input.addEventListener('focus', () => {
      input.parentElement?.classList.add('focused-glow');
    });
    input.addEventListener('blur', () => {
      input.parentElement?.classList.remove('focused-glow');
    });
  });
});
