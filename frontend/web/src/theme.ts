import { createTheme } from '@mui/material/styles';

export const UI_COLORS = {
  background: '#0D0F12', // Deep Obsidian
  surface: '#16191D',    // Dark Charcoal
  surfaceLight: '#1E2227',
  border: '#262B31',
  borderLight: '#3A4149',
  textPrimary: '#C9D1D9',   // Soft Silver
  textSecondary: '#8B949E', // Muted Slate
  textDisabled: '#484F58',
  accent: '#AE9E7B',        // Muted Bronze/Gold
  success: '#4D8B7A',       // Desaturated Sage Green
  danger: '#B35A5A',        // Muted Rose Red
  warning: '#D4A84B',       // Amber/Gold for medium confidence
  ma7: '#AE9E7B',           // Matching Bronze
  ma25: '#7B8AAE',          // Muted Steel Blue
  ma99: '#6B7280',          // Neutral Slate
  overlayBg: 'rgba(22, 25, 29, 0.92)',
};

const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: UI_COLORS.background,
      paper: UI_COLORS.surface,
    },
    primary: {
      main: UI_COLORS.accent,
    },
    text: {
      primary: UI_COLORS.textPrimary,
      secondary: UI_COLORS.textSecondary,
    },
    divider: UI_COLORS.border,
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    h6: {
      fontWeight: 500,
      letterSpacing: '0.02em',
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 4,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

export default theme;
