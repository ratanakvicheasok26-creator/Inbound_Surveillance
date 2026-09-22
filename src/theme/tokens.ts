/**
 * Digital Overwatch Design Tokens
 * High-contrast, minimalist theme built for Inbound Surveillance.
 */

export const DIGITAL_OVERWATCH_PALETTE = {
  absoluteBlack: {
    hex: "#000000",
    rgb: { r: 0, g: 0, b: 0 },
    rgbString: "0, 0, 0",
    usage: "The primary background color. Creates infinite depth and maximum contrast.",
  },
  surfaceCharcoal: {
    hex: "#121212",
    rgb: { r: 18, g: 18, b: 18 },
    rgbString: "18, 18, 18",
    usage: "Cards, panels, secondary surfaces, and modal backgrounds.",
  },
  surveillanceGreen: {
    hex: "#00FF66",
    rgb: { r: 0, g: 255, b: 102 },
    rgbString: "0, 255, 102",
    usage: "Primary brand action color, active HUD reticles, camera iris, live indicators.",
  },
  stealthGreen: {
    hex: "#0B833A",
    rgb: { r: 11, g: 131, b: 58 },
    rgbString: "11, 131, 58",
    usage: "Secondary darker green for inactive toggles, subtle borders, trailing glow shadows.",
  },
  crispWhite: {
    hex: "#FAFAFA",
    rgb: { r: 250, g: 250, b: 250 },
    rgbString: "250, 250, 250",
    usage: "Primary typography, mascot silhouette, and high-contrast lettering.",
  },
} as const;

export type DigitalOverwatchColorKey = keyof typeof DIGITAL_OVERWATCH_PALETTE;

export interface DesignTokenContract {
  color: {
    bg: {
      base: string;
      surface: string;
      surfaceElevated: string;
      surfaceHover: string;
    };
    brand: {
      primary: string;
      primaryGlow: string;
      stealth: string;
      stealthGlow: string;
    };
    text: {
      primary: string;
      muted: string;
      inverse: string;
    };
    border: {
      subtle: string;
      accent: string;
      focus: string;
    };
  };
  shadow: {
    laserGlow: string;
    stealthGlow: string;
    surfaceCard: string;
  };
}

export const designTokens: DesignTokenContract = {
  color: {
    bg: {
      base: DIGITAL_OVERWATCH_PALETTE.absoluteBlack.hex,
      surface: DIGITAL_OVERWATCH_PALETTE.surfaceCharcoal.hex,
      surfaceElevated: "#1A1A1A",
      surfaceHover: "#222222",
    },
    brand: {
      primary: DIGITAL_OVERWATCH_PALETTE.surveillanceGreen.hex,
      primaryGlow: "rgba(0, 255, 102, 0.45)",
      stealth: DIGITAL_OVERWATCH_PALETTE.stealthGreen.hex,
      stealthGlow: "rgba(11, 131, 58, 0.35)",
    },
    text: {
      primary: DIGITAL_OVERWATCH_PALETTE.crispWhite.hex,
      muted: "#8E9297",
      inverse: DIGITAL_OVERWATCH_PALETTE.absoluteBlack.hex,
    },
    border: {
      subtle: "rgba(255, 255, 255, 0.08)",
      accent: DIGITAL_OVERWATCH_PALETTE.stealthGreen.hex,
      focus: DIGITAL_OVERWATCH_PALETTE.surveillanceGreen.hex,
    },
  },
  shadow: {
    laserGlow: "0 0 12px rgba(0, 255, 102, 0.6), 0 0 24px rgba(11, 131, 58, 0.4)",
    stealthGlow: "0 0 16px rgba(11, 131, 58, 0.4)",
    surfaceCard: "0 4px 20px rgba(0, 0, 0, 0.8)",
  },
};
