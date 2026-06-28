import { Box, Stack, Typography } from "@mui/material";
import {
  getShapDiscreteColors,
  SHAP_DISCRETE_LABELS,
  type ShapValueMode,
} from "../utils/color";

type Props = {
  mode: ShapValueMode;
};

export function ShapLegend({ mode }: Props) {
  const discreteColors = getShapDiscreteColors();

  return (
    <Stack spacing={1}>
      <Typography variant="body2" sx={{ opacity: 0.8 }}>
        {mode === "discrete" ? "Discrete SHAP scale" : "Normalized SHAP scale"}
      </Typography>

      {mode === "discrete" ? (
        <>
          <Box
            sx={{
              height: 14,
              borderRadius: 999,
              overflow: "hidden",
              border: "1px solid rgba(255,255,255,0.12)",
              display: "grid",
              gridTemplateColumns: "repeat(5, 1fr)",
            }}
          >
            {discreteColors.map((color) => (
              <Box key={color} sx={{ backgroundColor: color }} />
            ))}
          </Box>

          <Stack direction="row">
            {SHAP_DISCRETE_LABELS.map((label) => (
              <Typography
                key={label}
                variant="caption"
                sx={{ flex: 1, textAlign: "center", opacity: 0.9 }}
              >
                {label}
              </Typography>
            ))}
          </Stack>
        </>
      ) : (
        <>
          <Box
            sx={{
              height: 14,
              borderRadius: 999,
              background:
                "linear-gradient(90deg, rgb(103,0,31) 0%, rgb(178,24,43) 18%, rgb(239,138,98) 36%, rgb(247,247,247) 50%, rgb(103,169,207) 64%, rgb(33,102,172) 82%, rgb(5,48,97) 100%)",
              border: "1px solid rgba(255,255,255,0.12)",
            }}
          />
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="caption">Negative</Typography>
            <Typography variant="caption">Zero</Typography>
            <Typography variant="caption">Positive</Typography>
          </Stack>
        </>
      )}
    </Stack>
  );
}
