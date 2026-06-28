import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#7c3aed",
    },
    secondary: {
      main: "#06b6d4",
    },
    background: {
      default: "#0b1020",
      paper: "rgba(18, 25, 46, 0.82)",
    },
  },
  shape: {
    borderRadius: 18,
  },
  typography: {
    fontFamily: `"Inter", "Segoe UI", "Roboto", sans-serif`,
    h4: {
      fontWeight: 800,
      letterSpacing: "-0.03em",
    },
    h6: {
      fontWeight: 700,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backdropFilter: "blur(18px)",
          border: "1px solid rgba(255,255,255,0.08)",
          boxShadow: "0 20px 80px rgba(0,0,0,0.35)",
        },
      },
    },
  },
});
