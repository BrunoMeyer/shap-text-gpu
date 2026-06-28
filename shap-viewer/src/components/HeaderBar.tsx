import { AppBar, Box, Toolbar, Typography } from "@mui/material";
import AutoGraphIcon from "@mui/icons-material/AutoGraph";

export function HeaderBar() {
  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        background: "linear-gradient(90deg, rgba(124,58,237,0.22), rgba(6,182,212,0.18))",
        backdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <Toolbar sx={{ gap: 1.5 }}>
        <AutoGraphIcon />
        <Box>
          <Typography variant="h6">SHAP Phrase Explorer</Typography>
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            Compare CPU and GPU token attributions side by side
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
}
