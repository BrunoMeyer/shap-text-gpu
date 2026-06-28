import {
  Card,
  CardContent,
  Chip,
  Divider,
  Grid,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { useState } from "react";
import type { ShapSample } from "../types/shap";
import { getGlobalMaxAbs, type ShapValueMode } from "../utils/color";
import { ShapLegend } from "./ShapLegend";
import { ShapPhrasePanel } from "./ShapPhrasePanel";

type Props = {
  sample: ShapSample | null;
};

export function ShapComparisonView({ sample }: Props) {
  const [shapValueMode, setShapValueMode] = useState<ShapValueMode>("discrete");
  const [showReferenceCard, setShowReferenceCard] = useState(false);

  if (!sample) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6">No sample selected</Typography>
        </CardContent>
      </Card>
    );
  }

  const maxAbsTokens = getGlobalMaxAbs(sample.tokens.map((t) => t.shap_value));
  const maxAbsGpuTokens = getGlobalMaxAbs(sample.shap_gpu_tokens.map((t) => t.shap_value));

  return (
    <Stack spacing={2.5}>
      <Card>
        <CardContent>
          <Stack spacing={2}>
            <Stack direction="row" flexWrap="wrap" gap={1}>
              <Chip label={sample.file_name} color="primary" />
              {Object.entries(sample.metadata).map(([key, value]) => (
                <Chip key={key} label={`${key}: ${String(value)}`} variant="outlined" size="small" />
              ))}
            </Stack>

            <div>
              <Typography variant="body2" sx={{ opacity: 0.7, mb: 0.5 }}>
                Full phrase
              </Typography>
              <Typography variant="body1" sx={{ fontSize: "1.05rem" }}>
                {sample.detokenized_full_text}
              </Typography>
            </div>

            <Divider />

            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="body2" sx={{ opacity: 0.8 }}>
                SHAP value display
              </Typography>
            </Stack>

            <Stack direction="row" justifyContent="space-between" alignItems="center" gap={2} flexWrap="wrap">
              <ToggleButtonGroup
                value={shapValueMode}
                exclusive
                size="small"
                onChange={(_, next) => {
                  if (next) setShapValueMode(next);
                }}
              >
                <ToggleButton value="discrete">Discrete</ToggleButton>
                <ToggleButton value="continuous">Continuous</ToggleButton>
              </ToggleButtonGroup>

              <ToggleButtonGroup
                value={showReferenceCard}
                exclusive
                size="small"
                onChange={(_, next) => {
                  if (next !== null) setShowReferenceCard(next);
                }}
              >
                <ToggleButton value={false}>Hide Reference</ToggleButton>
                <ToggleButton value={true}>Show Reference</ToggleButton>
              </ToggleButtonGroup>
            </Stack>

            <ShapLegend mode={shapValueMode} />
          </Stack>
        </CardContent>
      </Card>

      <Grid container spacing={2.5}>
        {showReferenceCard && (
          <Grid size={{ xs: 12, lg: 6 }}>
            <ShapPhrasePanel
              title="tokens"
              subtitle="Reference SHAP values"
              tokens={sample.tokens}
              maxAbs={maxAbsTokens}
              shapValueMode={shapValueMode}
            />
          </Grid>
        )}

        <Grid size={{ xs: 12, lg: showReferenceCard ? 6 : 12 }}>
          <ShapPhrasePanel
            title="shap_gpu_tokens"
            subtitle="GPU-generated SHAP values"
            tokens={sample.shap_gpu_tokens}
            maxAbs={maxAbsGpuTokens}
            shapValueMode={shapValueMode}
          />
        </Grid>
      </Grid>
    </Stack>
  );
}
