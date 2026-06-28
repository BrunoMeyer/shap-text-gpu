import { Card, CardContent, Chip, Stack, Typography } from "@mui/material";
import { motion } from "framer-motion";
import type { ShapToken } from "../types/shap";
import {
  formatShap,
  formatShapDiscrete,
  getTextColorForBackground,
  shapToColor,
  type ShapValueMode,
} from "../utils/color";

type Props = {
  title: string;
  subtitle?: string;
  tokens: ShapToken[];
  maxAbs: number;
  shapValueMode: ShapValueMode;
};

export function ShapPhrasePanel({
  title,
  subtitle,
  tokens,
  maxAbs,
  shapValueMode,
}: Props) {
  return (
    <Card sx={{ minHeight: 320 }}>
      <CardContent>
        <Stack spacing={2}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <div>
              <Typography variant="h6">{title}</Typography>
              {subtitle && (
                <Typography variant="body2" sx={{ opacity: 0.72 }}>
                  {subtitle}
                </Typography>
              )}
            </div>
            <Chip label={`max |SHAP| = ${maxAbs.toFixed(4)}`} size="small" color="secondary" />
          </Stack>

          <Stack
            direction="row"
            flexWrap="wrap"
            useFlexGap
            gap={1}
            sx={{
              alignItems: "flex-start",
              lineHeight: 2.2,
            }}
          >
            {tokens.map((token) => {
              const bg = shapToColor(token.shap_value, maxAbs, shapValueMode);
              const fg = getTextColorForBackground(token.shap_value, maxAbs, shapValueMode);

              return (
                <motion.div
                  key={`${title}-${token.idx}-${token.token_id}`}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.16, delay: token.idx * 0.008 }}
                >
                  <Stack
                    spacing={0.5}
                    sx={{
                      display: "inline-flex",
                      px: 1.1,
                      py: 0.8,
                      borderRadius: 2.5,
                      backgroundColor: bg,
                      color: fg,
                      border: "1px solid rgba(255,255,255,0.15)",
                      boxShadow: "0 6px 24px rgba(0,0,0,0.18)",
                    }}
                  >
                    <Typography variant="body1" sx={{ fontWeight: 700 }}>
                      {token.token_str}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{ opacity: 0.92, fontFamily: "monospace" }}
                    >
                      {shapValueMode === "discrete"
                        ? formatShapDiscrete(token.shap_value, maxAbs)
                        : formatShap(token.shap_value)}
                    </Typography>
                  </Stack>
                </motion.div>
              );
            })}
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
