import { useState } from "react";
import { Box, Button, Card, CardContent, CircularProgress, TextField, Typography, Alert } from "@mui/material";
import type { ShapSample } from "../types/shap";

type Props = {
  onResult: (sample: ShapSample | null) => void;
};

export function AdhocPanel({ onResult }: Props) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze() {
    setError(null);
    setLoading(true);
    onResult(null);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || res.statusText || "Analyze failed");
      }

      const data = await res.json();
      // Expect an array of samples from backend; pick first
      const sample = Array.isArray(data) ? data[0] ?? null : (data as ShapSample | null);
      onResult(sample);
    } catch (err: any) {
      setError(String(err?.message ?? err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Ad-hoc text analysis
        </Typography>

        <TextField
          label="Text to analyze"
          multiline
          minRows={3}
          fullWidth
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste text to run the SHAP pipeline..."
          sx={{ mb: 2 }}
        />

        {error && (
          <Box sx={{ mb: 2 }}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}

        <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
          <Button variant="contained" onClick={handleAnalyze} disabled={loading || !text.trim()}>
            Analyze
          </Button>
          {loading && <CircularProgress size={20} />}
        </Box>
      </CardContent>
    </Card>
  );
}

export default AdhocPanel;
