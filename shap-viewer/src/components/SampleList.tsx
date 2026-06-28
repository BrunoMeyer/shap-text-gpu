import {
  Card,
  CardContent,
  Chip,
  List,
  ListItemButton,
  ListItemText,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import type { ShapSample } from "../types/shap";

type Props = {
  samples: ShapSample[];
  search: string;
  onSearchChange: (value: string) => void;
  selectedIndex: number;
  onSelect: (index: number) => void;
};

export function SampleList({
  samples,
  search,
  onSearchChange,
  selectedIndex,
  onSelect,
}: Props) {
  return (
    <Card sx={{ height: "100%" }}>
      <CardContent>
        <Stack spacing={2}>
          <Typography variant="h6">Samples</Typography>

          <TextField
            fullWidth
            size="small"
            label="Search phrase or metadata"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
          />

          <Typography variant="body2" sx={{ opacity: 0.75 }}>
            {samples.length} result(s)
          </Typography>

          <List dense sx={{ maxHeight: "70vh", overflow: "auto", pr: 1 }}>
            {samples.map((sample, index) => {
              const sampleId = sample.metadata.sample ?? index;
              const target = sample.metadata.target ?? "-";
              const explainer = sample.metadata.explainer ?? "-";

              return (
                <ListItemButton
                  key={`${sample.file_name}-${index}`}
                  selected={selectedIndex === index}
                  onClick={() => onSelect(index)}
                  sx={{
                    mb: 1,
                    borderRadius: 3,
                    alignItems: "flex-start",
                    border: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  <ListItemText
                    primary={
                      <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                        <Chip label={`sample ${sampleId}`} size="small" color="primary" />
                        <Chip label={`target ${target}`} size="small" variant="outlined" />
                        <Chip label={String(explainer)} size="small" variant="outlined" />
                      </Stack>
                    }
                    secondary={
                      <Typography
                        variant="body2"
                        sx={{
                          mt: 1,
                          display: "-webkit-box",
                          WebkitLineClamp: 3,
                          WebkitBoxOrient: "vertical",
                          overflow: "hidden",
                          color: "rgba(255,255,255,0.72)",
                        }}
                      >
                        {sample.detokenized_full_text}
                      </Typography>
                    }
                  />
                </ListItemButton>
              );
            })}
          </List>
        </Stack>
      </CardContent>
    </Card>
  );
}
