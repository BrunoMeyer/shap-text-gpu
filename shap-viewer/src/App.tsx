import { useMemo, useState } from "react";
import {
  Alert,
  Box,
  CircularProgress,
  Container,
  Grid,
  Stack,
  Typography,
} from "@mui/material";
import { HeaderBar } from "./components/HeaderBar";
import { SampleList } from "./components/SampleList";
import { ShapComparisonView } from "./components/ShapComparisonView";
import { AdhocPanel } from "./components/AdhocPanel";
import { useSamples } from "./hooks/useSamples";
import type { ShapSample } from "./types/shap";

function matchesSearch(sample: ShapSample, search: string) {
  const q = search.trim().toLowerCase();
  if (!q) return true;

  const meta = Object.entries(sample.metadata)
    .map(([k, v]) => `${k} ${String(v)}`)
    .join(" ")
    .toLowerCase();

  const phrase = sample.detokenized_full_text.toLowerCase();
  const fileName = sample.file_name.toLowerCase();

  return phrase.includes(q) || meta.includes(q) || fileName.includes(q);
}

export default function App() {
  const { data, isLoading, isError, error } = useSamples();
  const [search, setSearch] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [adhocSample, setAdhocSample] = useState<ShapSample | null>(null);

  const filteredSamples = useMemo(() => {
    return (data ?? []).filter((sample) => matchesSearch(sample, search));
  }, [data, search]);

  const selectedSample = filteredSamples[selectedIndex] ?? null;
  const activeSample = adhocSample ?? selectedSample;

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background:
          "radial-gradient(circle at top left, rgba(124,58,237,0.22), transparent 30%), radial-gradient(circle at top right, rgba(6,182,212,0.16), transparent 30%), linear-gradient(180deg, #0b1020 0%, #11172b 100%)",
      }}
    >
      <HeaderBar />

      <Container maxWidth="xl" sx={{ py: 3 }}>
        <Stack spacing={2.5}>
          <Box>
            <Typography variant="h4" gutterBottom>
              SHAP Comparison Dashboard
            </Typography>
            <Typography variant="body1" sx={{ opacity: 0.78 }}>
              Search samples, inspect phrase-level attributions, and compare standard tokens
              against GPU SHAP values using a normalized color scale.
            </Typography>
          </Box>

          {isLoading && (
            <Box sx={{ display: "flex", justifyContent: "center", py: 10 }}>
              <CircularProgress />
            </Box>
          )}

          {isError && (
            <Alert severity="error">
              {(error as Error)?.message || "Failed to load data"}
            </Alert>
          )}

          {!isLoading && !isError && (
            <Grid container spacing={2.5}>
              <Grid size={{ xs: 12, lg: 4, xl: 3 }}>
                <SampleList
                  samples={filteredSamples}
                  search={search}
                  onSearchChange={(value) => {
                    setSearch(value);
                    setSelectedIndex(0);
                  }}
                  selectedIndex={selectedIndex}
                  onSelect={setSelectedIndex}
                />
              </Grid>

              <Grid size={{ xs: 12, lg: 8, xl: 9 }}>
                <AdhocPanel
                  onResult={(sample) => {
                    setAdhocSample(sample);
                  }}
                />

                <Box sx={{ height: 16 }} />

                <ShapComparisonView sample={activeSample} />
              </Grid>
            </Grid>
          )}
        </Stack>
      </Container>
    </Box>
  );
}
