export type ShapToken = {
  idx: number;
  token_id: number;
  token_str: string;
  shap_value: number;
};

export type SampleMetadata = Record<string, string | number | boolean | null>;

export type ShapSample = {
  file_name: string;
  metadata: SampleMetadata;
  detokenized_full_text: string;
  tokens: ShapToken[];
  shap_gpu_tokens: ShapToken[];
};

export type SamplesResponse = ShapSample[];
